//! The 3D scene: hand-written WebGL2, no framework.
//!
//! World frame is the simulator's — x, y horizontal, z up, metres. It draws a ground grid, the
//! true and estimated trajectories as trails, the estimate as a lit quadrotor with a faint ghost
//! at ground truth, the fixed UWB beacons with range lines when they are ranging, the downward
//! LiDAR beam, and — the point of the whole thing — the filter's position uncertainty as a
//! covariance ellipsoid: a unit sphere pushed through `k·L`, where `L Lᵀ` is the 3×3 position
//! covariance. When the filter is confident it is a tight bead on the path; when GPS drops it
//! swells to metres, in real units, exactly as the covariance says it should.

import {
  ellipsoidTransform,
  lookAt,
  modelMatrix,
  multiply,
  normalMatrix,
  orbitEye,
  perspective,
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

// Lit mesh (the quadrotor): a single directional light in world space.
const LIT_VERT = `#version 300 es
in vec3 aPos;
in vec3 aNormal;
uniform mat4 uMVP;
uniform mat3 uNormal;
out vec3 vN;
void main() { vN = uNormal * aNormal; gl_Position = uMVP * vec4(aPos, 1.0); }`

const LIT_FRAG = `#version 300 es
precision highp float;
in vec3 vN;
uniform vec4 uColor;
uniform vec3 uLight;
out vec4 frag;
void main() {
  float d = max(dot(normalize(vN), normalize(uLight)), 0.0);
  frag = vec4(uColor.rgb * (0.32 + 0.68 * d), uColor.a);
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
  /** Snapshot from eskf-wasm: est p(3) v(3) q(4), truth p(3) q(4), cov(9), metrics, biases,
   *  sensor pulses at [36..42] = gps,baro,mag,lidar,uwb,flow. */
  snapshot: Float32Array
}

const CENTER: Vec3 = [0, 0, 12]
const ELLIPSOID_K = 2.795 // sqrt of the χ²₃ 95% quantile — the 95% position ellipsoid
const DRONE_SCALE = 2.3
const LIGHT: Vec3 = [0.4, 0.5, 1.0]
const P_LIDAR = 36 + 3
const P_UWB = 36 + 4

export class Scene {
  private readonly gl: WebGL2RenderingContext
  private readonly line: Prog
  private readonly point: Prog
  private readonly lit: Prog
  private readonly ell: Prog

  private readonly grid: { vao: WebGLVertexArrayObject; count: number }
  private readonly estBuf: Dyn
  private readonly truthBuf: Dyn
  private readonly rayBuf: Dyn
  private readonly frame: Mesh
  private readonly rotors: Mesh
  private readonly sphere: { vao: WebGLVertexArrayObject; index: number }
  private readonly rings: { vao: WebGLVertexArrayObject; count: number }

  private readonly beacons: Vec3[]
  private readonly beaconPoints: { vao: WebGLVertexArrayObject; count: number }
  private readonly beaconStalks: { vao: WebGLVertexArrayObject; count: number }

  private az = 0.9
  private el = 0.5
  private dist = 150
  private dragging = false
  private lastX = 0
  private lastY = 0

  constructor(private readonly canvas: HTMLCanvasElement, beacons: Float32Array) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false })
    if (!gl) throw new Error('WebGL2 is unavailable in this browser.')
    this.gl = gl

    this.line = makeProg(gl, LINE_VERT, LINE_FRAG, ['uMVP', 'uColor'])
    this.point = makeProg(gl, POINT_VERT, POINT_FRAG, ['uMVP', 'uColor', 'uSize'])
    this.lit = makeProg(gl, LIT_VERT, LIT_FRAG, ['uMVP', 'uNormal', 'uColor', 'uLight'])
    this.ell = makeProg(gl, ELL_VERT, ELL_FRAG, ['uMVP', 'uColor', 'uL', 'uCenter'])

    this.grid = this.buildGrid()
    this.estBuf = makeDyn(gl, 0, 3, 3)
    this.truthBuf = makeDyn(gl, 0, 3, 3)
    this.rayBuf = makeDyn(gl, 0, 3, 3)
    this.frame = this.buildFrameMesh()
    this.rotors = this.buildRotorMesh()
    this.sphere = this.buildSphere(24, 16)
    this.rings = this.buildRings(64)

    this.beacons = []
    for (let i = 0; i + 2 < beacons.length; i += 3) {
      this.beacons.push([beacons[i]!, beacons[i + 1]!, beacons[i + 2]!])
    }
    this.beaconPoints = {
      vao: staticVao(gl, new Float32Array(this.beacons.flat())),
      count: this.beacons.length,
    }
    const stalks: number[] = []
    for (const b of this.beacons) stalks.push(b[0], b[1], b[2], b[0], b[1], 0)
    this.beaconStalks = { vao: staticVao(gl, new Float32Array(stalks)), count: stalks.length / 3 }

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
    const vp = multiply(proj, view)

    // Grid + beacons + stalks.
    this.drawLines(this.grid.vao, this.grid.count, vp, [0.22, 0.28, 0.4, 0.55], gl.LINES)
    this.drawLines(this.beaconStalks.vao, this.beaconStalks.count, vp, [0.7, 0.45, 1.0, 0.4], gl.LINES)
    this.drawPoints(this.beaconPoints.vao, this.beaconPoints.count, vp, [0.78, 0.5, 1.0, 1], 10)

    const s = frame.snapshot
    const estP: Vec3 = [s[0]!, s[1]!, s[2]!]
    const estQ: [number, number, number, number] = [s[6]!, s[7]!, s[8]!, s[9]!]
    const truthP: Vec3 = [s[10]!, s[11]!, s[12]!]
    const truthQ: [number, number, number, number] = [s[13]!, s[14]!, s[15]!, s[16]!]

    // Trails.
    upload(gl, this.truthBuf, frame.truthTrail)
    this.drawLines(this.truthBuf.vao, frame.truthTrail.length / 3, vp, [0.35, 0.9, 0.55, 0.95], gl.LINE_STRIP)
    upload(gl, this.estBuf, frame.estTrail)
    this.drawLines(this.estBuf.vao, frame.estTrail.length / 3, vp, [0.45, 0.75, 1.0, 0.95], gl.LINE_STRIP)

    // UWB range lines (when ranging) and the LiDAR down-beam.
    const uwbPulse = s[P_UWB]!
    if (uwbPulse > 0.05) {
      const segs: number[] = []
      for (const b of this.beacons) segs.push(estP[0], estP[1], estP[2], b[0], b[1], b[2])
      upload(gl, this.rayBuf, new Float32Array(segs))
      this.drawLines(this.rayBuf.vao, segs.length / 3, vp, [0.78, 0.5, 1.0, 0.35 + 0.4 * uwbPulse], gl.LINES)
    }
    const lidarPulse = s[P_LIDAR]!
    if (lidarPulse > 0.05) {
      const ax = normalMatrix(estQ) // columns are body axes; body z is columns[2]
      const upZ: Vec3 = [ax[6]!, ax[7]!, ax[8]!]
      if (upZ[2] > 0.05 && estP[2] > 0) {
        const t = estP[2] / upZ[2]
        const hit: Vec3 = [estP[0] - upZ[0] * t, estP[1] - upZ[1] * t, estP[2] - upZ[2] * t]
        upload(gl, this.rayBuf, new Float32Array([estP[0], estP[1], estP[2], hit[0], hit[1], hit[2]]))
        this.drawLines(this.rayBuf.vao, 2, vp, [1.0, 0.7, 0.3, 0.3 + 0.4 * lidarPulse], gl.LINES)
      }
    }

    // The ghost (truth) quadrotor, then the estimate, then the ellipsoid.
    gl.depthMask(false)
    this.drawDrone(vp, truthP, truthQ, [0.4, 0.95, 0.55], [0.5, 0.8, 0.6], 0.4)
    gl.depthMask(true)
    this.drawDrone(vp, estP, estQ, [0.5, 0.72, 1.0], [0.75, 0.85, 1.0], 1.0)

    const kL = ellipsoidTransform(s.subarray(17, 26), ELLIPSOID_K)
    this.drawEllipsoid(vp, estP, kL)
  }

  private drawDrone(vp: Mat4, p: Vec3, q: [number, number, number, number], body: number[], rotor: number[], alpha: number) {
    const gl = this.gl
    const model = modelMatrix(p, q, DRONE_SCALE)
    const mvp = multiply(vp, model)
    const nrm = normalMatrix(q)
    gl.useProgram(this.lit.program)
    gl.uniformMatrix4fv(this.lit.u.uMVP!, false, mvp)
    gl.uniformMatrix3fv(this.lit.u.uNormal!, false, nrm)
    gl.uniform3fv(this.lit.u.uLight!, LIGHT)
    gl.uniform4fv(this.lit.u.uColor!, [...body, alpha])
    gl.bindVertexArray(this.frame.vao)
    gl.drawArrays(gl.TRIANGLES, 0, this.frame.count)
    gl.uniform4fv(this.lit.u.uColor!, [...rotor, alpha])
    gl.bindVertexArray(this.rotors.vao)
    gl.drawArrays(gl.TRIANGLES, 0, this.rotors.count)
    gl.bindVertexArray(null)
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
    gl.uniform1f(this.point.u.uSize!, size * Math.min(window.devicePixelRatio || 1, 2))
    gl.bindVertexArray(vao)
    gl.drawArrays(gl.POINTS, 0, count)
    gl.bindVertexArray(null)
  }

  private drawEllipsoid(mvp: Mat4, center: Vec3, kL: Float32Array) {
    const gl = this.gl
    gl.useProgram(this.ell.program)
    gl.uniformMatrix4fv(this.ell.u.uMVP!, false, mvp)
    gl.uniformMatrix3fv(this.ell.u.uL!, false, kL)
    gl.uniform3fv(this.ell.u.uCenter!, center)
    gl.depthMask(false)
    gl.uniform4fv(this.ell.u.uColor!, [0.95, 0.65, 0.25, 0.16])
    gl.bindVertexArray(this.sphere.vao)
    gl.drawElements(gl.TRIANGLES, this.sphere.index, gl.UNSIGNED_SHORT, 0)
    gl.depthMask(true)
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
    verts.push(0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 15)
    return { vao: staticVao(gl, new Float32Array(verts)), count: verts.length / 3 }
  }

  private buildFrameMesh(): Mesh {
    const v: number[] = []
    // Body, four arms (a +-config), and a forward nose spike so heading is legible.
    pushBox(v, [0, 0, 0], [0.34, 0.34, 0.13])
    pushBox(v, [0.55, 0, 0], [0.55, 0.05, 0.04])
    pushBox(v, [-0.55, 0, 0], [0.55, 0.05, 0.04])
    pushBox(v, [0, 0.55, 0], [0.05, 0.55, 0.04])
    pushBox(v, [0, -0.55, 0], [0.05, 0.55, 0.04])
    pushBox(v, [0.78, 0, 0.1], [0.28, 0.05, 0.03]) // nose
    return litVao(this.gl, new Float32Array(v))
  }

  private buildRotorMesh(): Mesh {
    const v: number[] = []
    for (const c of [[1.05, 0, 0.11], [-1.05, 0, 0.11], [0, 1.05, 0.11], [0, -1.05, 0.11]] as Vec3[]) {
      pushDisc(v, c, 0.42, 0.03, 16)
    }
    return litVao(this.gl, new Float32Array(v))
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
      push([c0, s0, 0], [c1, s1, 0])
      push([0, c0, s0], [0, c1, s1])
      push([c0, 0, s0], [c1, 0, s1])
    }
    return { vao: staticVao(this.gl, new Float32Array(pts)), count: pts.length / 3 }
  }
}

interface Mesh {
  vao: WebGLVertexArrayObject
  count: number
}

interface Dyn {
  vao: WebGLVertexArrayObject
  vbo: WebGLBuffer
  cap: number
}

/** Push an axis-aligned box (centre, half-extents) as 12 triangles with flat normals. */
function pushBox(out: number[], c: Vec3, h: Vec3) {
  const faces: [Vec3, Vec3, Vec3][] = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
  ]
  for (const [n, u, w] of faces) {
    const centre: Vec3 = [c[0] + n[0] * h[0], c[1] + n[1] * h[1], c[2] + n[2] * h[2]]
    const uu: Vec3 = [u[0] * h[0], u[1] * h[1], u[2] * h[2]]
    const ww: Vec3 = [w[0] * h[0], w[1] * h[1], w[2] * h[2]]
    const corner = (su: number, sw: number): Vec3 => [
      centre[0] + su * uu[0] + sw * ww[0],
      centre[1] + su * uu[1] + sw * ww[1],
      centre[2] + su * uu[2] + sw * ww[2],
    ]
    const a = corner(-1, -1), b = corner(1, -1), d = corner(1, 1), e = corner(-1, 1)
    for (const p of [a, b, d, a, d, e]) out.push(p[0], p[1], p[2], n[0], n[1], n[2])
  }
}

/** Push a short cylinder (rotor disc): top, bottom, and a side ring. */
function pushDisc(out: number[], c: Vec3, r: number, halfH: number, segs: number) {
  const top = c[2] + halfH
  const bot = c[2] - halfH
  for (let i = 0; i < segs; i++) {
    const a0 = (i / segs) * 2 * Math.PI
    const a1 = ((i + 1) / segs) * 2 * Math.PI
    const x0 = c[0] + r * Math.cos(a0), y0 = c[1] + r * Math.sin(a0)
    const x1 = c[0] + r * Math.cos(a1), y1 = c[1] + r * Math.sin(a1)
    // top
    out.push(c[0], c[1], top, 0, 0, 1, x0, y0, top, 0, 0, 1, x1, y1, top, 0, 0, 1)
    // bottom
    out.push(c[0], c[1], bot, 0, 0, -1, x1, y1, bot, 0, 0, -1, x0, y0, bot, 0, 0, -1)
    // side
    const nx0 = Math.cos(a0), ny0 = Math.sin(a0), nx1 = Math.cos(a1), ny1 = Math.sin(a1)
    out.push(x0, y0, bot, nx0, ny0, 0, x1, y1, bot, nx1, ny1, 0, x1, y1, top, nx1, ny1, 0)
    out.push(x0, y0, bot, nx0, ny0, 0, x1, y1, top, nx1, ny1, 0, x0, y0, top, nx0, ny0, 0)
  }
}

function makeDyn(gl: WebGL2RenderingContext, loc: number, size: number, stride: number): Dyn {
  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)
  const vbo = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
  gl.enableVertexAttribArray(loc)
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride * 4, 0)
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

/** Interleaved position(3) + normal(3) VAO for the lit meshes. */
function litVao(gl: WebGL2RenderingContext, data: Float32Array): Mesh {
  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)
  const vbo = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0)
  gl.enableVertexAttribArray(1)
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12)
  gl.bindVertexArray(null)
  return { vao, count: data.length / 6 }
}

function makeProg(gl: WebGL2RenderingContext, vs: string, fs: string, uniforms: string[]): Prog {
  const program = gl.createProgram()!
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs))
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs))
  gl.bindAttribLocation(program, 0, 'aPos')
  gl.bindAttribLocation(program, 1, 'aNormal')
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
