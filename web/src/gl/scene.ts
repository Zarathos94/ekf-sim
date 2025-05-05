//! The 3D scene: hand-written WebGL2, no framework.
//!
//! World frame is the simulator's — x, y horizontal, z up, metres. It draws a ground grid, the
//! true and estimated trajectories as trails, the estimate's orientation as a body-axis triad,
//! and — the point of the whole thing — the filter's position uncertainty as a covariance
//! ellipsoid: a unit sphere pushed through `k·L`, where `L Lᵀ` is the 3×3 position covariance.
//! When the filter is confident the ellipsoid is a tight bead on the path; when GPS drops it
//! swells to metres, in real units, exactly as the covariance says it should.

import {
  ellipsoidTransform,
  lookAt,
  multiply,
  orbitEye,
  perspective,
  quatAxes,
  type Mat4,
  type Vec3,
} from '../mat.js'

const LINE_VERT = `#version 300 es
in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`

const LINE_FRAG = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 frag;
void main() { frag = uColor; }`

const POINT_VERT = `#version 300 es
in vec3 aPos;
uniform mat4 uMVP;
uniform float uSize;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); gl_PointSize = uSize; }`

const POINT_FRAG = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 frag;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  if (dot(c, c) > 1.0) discard;
  frag = uColor;
}`

// The ellipsoid: each unit-sphere vertex is placed at center + kL·v.
const ELL_VERT = `#version 300 es
in vec3 aPos;
uniform mat4 uMVP;
uniform mat3 uL;
uniform vec3 uCenter;
out vec3 vN;
void main() {
  vN = normalize(aPos);
  vec3 world = uCenter + uL * aPos;
  gl_Position = uMVP * vec4(world, 1.0);
}`

const ELL_FRAG = `#version 300 es
precision highp float;
in vec3 vN;
uniform vec4 uColor;
out vec4 frag;
void main() {
  float shade = 0.55 + 0.45 * abs(vN.z);
  frag = vec4(uColor.rgb * shade, uColor.a);
}`

interface Prog {
  program: WebGLProgram
  u: Record<string, WebGLUniformLocation | null>
}

export interface FrameData {
  estTrail: Float32Array
  truthTrail: Float32Array
  /** Snapshot layout from eskf-wasm: est p(3) v(3) q(4), truth p(3) q(4), cov(9), metrics… */
  snapshot: Float32Array
}

const CENTER: Vec3 = [0, 0, 12]
const ELLIPSOID_K = 2.795 // sqrt of the χ²₃ 95% quantile — the 95% position ellipsoid

export class Scene {
  private readonly gl: WebGL2RenderingContext
  private readonly line: Prog
  private readonly point: Prog
  private readonly ell: Prog

  private readonly grid: { vao: WebGLVertexArrayObject; count: number }
  private readonly estBuf: Dyn
  private readonly truthBuf: Dyn
  private readonly triadBuf: Dyn
  private readonly markerBuf: Dyn
  private readonly sphere: { vao: WebGLVertexArrayObject; index: number }
  private readonly rings: { vao: WebGLVertexArrayObject; count: number }

  private az = 0.9
  private el = 0.5
  private dist = 150
  private dragging = false
  private lastX = 0
  private lastY = 0

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false })
    if (!gl) throw new Error('WebGL2 is unavailable in this browser.')
    this.gl = gl

    this.line = makeProg(gl, LINE_VERT, LINE_FRAG, ['uMVP', 'uColor'])
    this.point = makeProg(gl, POINT_VERT, POINT_FRAG, ['uMVP', 'uColor', 'uSize'])
    this.ell = makeProg(gl, ELL_VERT, ELL_FRAG, ['uMVP', 'uColor', 'uL', 'uCenter'])

    this.grid = this.buildGrid()
    this.estBuf = makeDyn(gl)
    this.truthBuf = makeDyn(gl)
    this.triadBuf = makeDyn(gl)
    this.markerBuf = makeDyn(gl)
    this.sphere = this.buildSphere(24, 16)
    this.rings = this.buildRings(64)

    this.installControls()
    this.resize()
  }

  private installControls() {
    const c = this.canvas
    c.addEventListener('pointerdown', (e) => {
      this.dragging = true
      this.lastX = e.clientX
      this.lastY = e.clientY
      c.setPointerCapture(e.pointerId)
    })
    c.addEventListener('pointerup', (e) => {
      this.dragging = false
      c.releasePointerCapture(e.pointerId)
    })
    c.addEventListener('pointermove', (e) => {
      if (!this.dragging) return
      this.az -= (e.clientX - this.lastX) * 0.006
      this.el = clamp(this.el + (e.clientY - this.lastY) * 0.006, -1.45, 1.45)
      this.lastX = e.clientX
      this.lastY = e.clientY
    })
    c.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault()
        this.dist = clamp(this.dist * Math.exp(e.deltaY * 0.001), 40, 400)
      },
      { passive: false },
    )
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr))
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr))
  }

  render(frame: FrameData) {
    const gl = this.gl
    // Self-heal the backing store if layout changed.
    const rect = this.canvas.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    if (Math.round(rect.width * dpr) !== this.canvas.width && rect.width > 0) this.resize()

    gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    gl.clearColor(0.03, 0.04, 0.06, 1)
    gl.enable(gl.DEPTH_TEST)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

    const aspect = this.canvas.width / Math.max(1, this.canvas.height)
    const proj = perspective((50 * Math.PI) / 180, aspect, 0.5, 2000)
    const eye = orbitEye(this.az, this.el, this.dist, CENTER)
    const view = lookAt(eye, CENTER, [0, 0, 1])
    const mvp = multiply(proj, view)

    // Grid.
    this.drawLines(this.grid.vao, this.grid.count, mvp, [0.22, 0.28, 0.4, 0.55], gl.LINES)

    // Trails.
    const s = frame.snapshot
    upload(gl, this.truthBuf, frame.truthTrail)
    this.drawLines(this.truthBuf.vao, frame.truthTrail.length / 3, mvp, [0.35, 0.9, 0.55, 0.95], gl.LINE_STRIP)
    upload(gl, this.estBuf, frame.estTrail)
    this.drawLines(this.estBuf.vao, frame.estTrail.length / 3, mvp, [0.45, 0.75, 1.0, 0.95], gl.LINE_STRIP)

    // Orientation triad at the estimate + position markers.
    const estP: Vec3 = [s[0]!, s[1]!, s[2]!]
    const estQ: [number, number, number, number] = [s[6]!, s[7]!, s[8]!, s[9]!]
    const truthP: Vec3 = [s[10]!, s[11]!, s[12]!]
    this.drawTriad(mvp, estP, estQ)
    upload(gl, this.markerBuf, new Float32Array([...truthP]))
    this.drawPoints(this.markerBuf.vao, 1, mvp, [0.4, 1.0, 0.6, 1], 9)
    upload(gl, this.markerBuf, new Float32Array([...estP]))
    this.drawPoints(this.markerBuf.vao, 1, mvp, [0.55, 0.8, 1.0, 1], 11)

    // Covariance ellipsoid, centred on the estimate.
    const cov = s.subarray(17, 26)
    const kL = ellipsoidTransform(cov, ELLIPSOID_K)
    this.drawEllipsoid(mvp, estP, kL)
  }

  private drawLines(vao: WebGLVertexArrayObject, count: number, mvp: Mat4, color: number[], mode: number) {
    if (count <= 0) return
    const gl = this.gl
    gl.useProgram(this.line.program)
    gl.uniformMatrix4fv(this.line.u.uMVP!, false, mvp)
    gl.uniform4fv(this.line.u.uColor!, color)
    gl.bindVertexArray(vao)
    gl.drawArrays(mode, 0, count)
    gl.bindVertexArray(null)
  }

  private drawPoints(vao: WebGLVertexArrayObject, count: number, mvp: Mat4, color: number[], size: number) {
    const gl = this.gl
    gl.useProgram(this.point.program)
    gl.uniformMatrix4fv(this.point.u.uMVP!, false, mvp)
    gl.uniform4fv(this.point.u.uColor!, color)
    gl.uniform1f(this.point.u.uSize!, size * (Math.min(window.devicePixelRatio || 1, 2)))
    gl.bindVertexArray(vao)
    gl.drawArrays(gl.POINTS, 0, count)
    gl.bindVertexArray(null)
  }

  private drawTriad(mvp: Mat4, p: Vec3, q: [number, number, number, number]) {
    const axes = quatAxes(q)
    const len = 5
    const seg = (dir: Vec3) => [p[0], p[1], p[2], p[0] + dir[0] * len, p[1] + dir[1] * len, p[2] + dir[2] * len]
    for (const [dir, color] of [
      [axes.x, [1.0, 0.4, 0.4, 1]],
      [axes.y, [0.4, 1.0, 0.5, 1]],
      [axes.z, [0.5, 0.6, 1.0, 1]],
    ] as [Vec3, number[]][]) {
      upload(this.gl, this.triadBuf, new Float32Array(seg(dir)))
      this.drawLines(this.triadBuf.vao, 2, mvp, color, this.gl.LINES)
    }
  }

  private drawEllipsoid(mvp: Mat4, center: Vec3, kL: Float32Array) {
    const gl = this.gl
    gl.useProgram(this.ell.program)
    gl.uniformMatrix4fv(this.ell.u.uMVP!, false, mvp)
    gl.uniformMatrix3fv(this.ell.u.uL!, false, kL)
    gl.uniform3fv(this.ell.u.uCenter!, center)

    // Translucent shell (no depth write, so the path shows through).
    gl.depthMask(false)
    gl.uniform4fv(this.ell.u.uColor!, [0.95, 0.65, 0.25, 0.16])
    gl.bindVertexArray(this.sphere.vao)
    gl.drawElements(gl.TRIANGLES, this.sphere.index, gl.UNSIGNED_SHORT, 0)
    gl.depthMask(true)

    // Three principal rings for a crisp outline.
    gl.uniform4fv(this.ell.u.uColor!, [1.0, 0.75, 0.35, 0.9])
    gl.bindVertexArray(this.rings.vao)
    gl.drawArrays(gl.LINES, 0, this.rings.count)
    gl.bindVertexArray(null)
  }

  // --- static geometry ---

  private buildGrid() {
    const gl = this.gl
    const verts: number[] = []
    const ext = 60
    const step = 10
    for (let x = -ext; x <= ext; x += step) {
      verts.push(x, -ext, 0, x, ext, 0)
      verts.push(-ext, x, 0, ext, x, 0)
    }
    // World axes from the origin.
    verts.push(0, 0, 0, 15, 0, 0)
    verts.push(0, 0, 0, 0, 15, 0)
    verts.push(0, 0, 0, 0, 0, 15)
    return { vao: staticVao(gl, new Float32Array(verts)), count: verts.length / 3 }
  }

  private buildSphere(lon: number, lat: number) {
    const gl = this.gl
    const pos: number[] = []
    for (let i = 0; i <= lat; i++) {
      const theta = (i / lat) * Math.PI
      const st = Math.sin(theta)
      const ct = Math.cos(theta)
      for (let j = 0; j <= lon; j++) {
        const phi = (j / lon) * 2 * Math.PI
        pos.push(st * Math.cos(phi), st * Math.sin(phi), ct)
      }
    }
    const idx: number[] = []
    for (let i = 0; i < lat; i++) {
      for (let j = 0; j < lon; j++) {
        const a = i * (lon + 1) + j
        const b = a + lon + 1
        idx.push(a, b, a + 1, a + 1, b, b + 1)
      }
    }
    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)
    const ibo = gl.createBuffer()!
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW)
    gl.bindVertexArray(null)
    return { vao, index: idx.length }
  }

  private buildRings(seg: number) {
    const pts: number[] = []
    const push = (a: Vec3, b: Vec3) => pts.push(a[0], a[1], a[2], b[0], b[1], b[2])
    for (let i = 0; i < seg; i++) {
      const t0 = (i / seg) * 2 * Math.PI
      const t1 = ((i + 1) / seg) * 2 * Math.PI
      const c0 = Math.cos(t0), s0 = Math.sin(t0), c1 = Math.cos(t1), s1 = Math.sin(t1)
      push([c0, s0, 0], [c1, s1, 0]) // xy
      push([0, c0, s0], [0, c1, s1]) // yz
      push([c0, 0, s0], [c1, 0, s1]) // xz
    }
    return { vao: staticVao(this.gl, new Float32Array(pts)), count: pts.length / 3 }
  }
}

interface Dyn {
  vao: WebGLVertexArrayObject
  vbo: WebGLBuffer
  cap: number
}

function makeDyn(gl: WebGL2RenderingContext): Dyn {
  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)
  const vbo = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)
  gl.bindVertexArray(null)
  return { vao, vbo, cap: 0 }
}

function upload(gl: WebGL2RenderingContext, d: Dyn, data: Float32Array) {
  gl.bindBuffer(gl.ARRAY_BUFFER, d.vbo)
  if (data.byteLength > d.cap) {
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW)
    d.cap = data.byteLength
  } else {
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data)
  }
}

function staticVao(gl: WebGL2RenderingContext, data: Float32Array): WebGLVertexArrayObject {
  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)
  const vbo = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)
  gl.bindVertexArray(null)
  return vao
}

function makeProg(gl: WebGL2RenderingContext, vs: string, fs: string, uniforms: string[]): Prog {
  const program = gl.createProgram()!
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs))
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs))
  gl.bindAttribLocation(program, 0, 'aPos')
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`link failed: ${gl.getProgramInfoLog(program)}`)
  }
  const u: Record<string, WebGLUniformLocation | null> = {}
  for (const name of uniforms) u[name] = gl.getUniformLocation(program, name)
  return { program, u }
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)!
  gl.shaderSource(shader, src)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`shader compile failed: ${gl.getShaderInfoLog(shader)}`)
  }
  return shader
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x
}
