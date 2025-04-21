//! A minimal column-major 4×4 / 3×3 kit for the scene camera and the covariance ellipsoid.
//!
//! No matrix library for a handful of operations, matching the hand-written renderer. The world
//! frame is the simulator's: x, y horizontal and z up, metres.

export type Mat4 = Float32Array
export type Vec3 = [number, number, number]

export function identity(): Mat4 {
  const m = new Float32Array(16)
  m[0] = m[5] = m[10] = m[15] = 1
  return m
}

/** Right-handed perspective projection. `fovY` in radians. */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2)
  const m = new Float32Array(16)
  m[0] = f / aspect
  m[5] = f
  m[10] = (far + near) / (near - far)
  m[11] = -1
  m[14] = (2 * far * near) / (near - far)
  return m
}

/** Right-handed look-at view matrix. */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, target))
  const x = normalize(cross(up, z))
  const y = cross(z, x)
  const m = new Float32Array(16)
  m[0] = x[0]; m[1] = y[0]; m[2] = z[0]; m[3] = 0
  m[4] = x[1]; m[5] = y[1]; m[6] = z[1]; m[7] = 0
  m[8] = x[2]; m[9] = y[2]; m[10] = z[2]; m[11] = 0
  m[12] = -dot(x, eye); m[13] = -dot(y, eye); m[14] = -dot(z, eye); m[15] = 1
  return m
}

/** `a · b`, column-major. */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16)
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[k * 4 + r]! * b[c * 4 + k]!
      out[c * 4 + r] = s
    }
  }
  return out
}

export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export function normalize(a: Vec3): Vec3 {
  const n = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / n, a[1] / n, a[2] / n]
}

/**
 * Camera position at azimuth/elevation (radians) and radius, looking at `center`. World up is
 * +z, so elevation tilts toward the pole.
 */
export function orbitEye(azimuth: number, elevation: number, radius: number, center: Vec3): Vec3 {
  const ce = Math.cos(elevation)
  return [
    center[0] + radius * ce * Math.cos(azimuth),
    center[1] + radius * ce * Math.sin(azimuth),
    center[2] + radius * Math.sin(elevation),
  ]
}

/**
 * The three body axes of a Hamilton quaternion `[w,x,y,z]`, expressed in the world frame — the
 * columns of its rotation matrix. Used to draw the orientation triad.
 */
export function quatAxes(q: [number, number, number, number]): { x: Vec3; y: Vec3; z: Vec3 } {
  const [w, x, y, z] = q
  return {
    x: [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    y: [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    z: [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  }
}

/**
 * Lower-triangular Cholesky factor of a symmetric 3×3 covariance (row-major in), returned as a
 * column-major `mat3` scaled by `k`. Transforming a unit sphere by `k·L` yields the `k`-σ
 * uncertainty ellipsoid, since `L Lᵀ = P`. Diagonal terms are floored so a momentarily
 * non-positive-definite covariance still produces a (degenerate but finite) ellipsoid.
 */
export function ellipsoidTransform(cov: Float32Array | number[], k: number): Float32Array {
  const eps = 1e-9
  const p00 = cov[0]!, p01 = cov[1]!, p02 = cov[2]!
  const p11 = cov[4]!, p12 = cov[5]!, p22 = cov[8]!
  const l00 = Math.sqrt(Math.max(p00, eps))
  const l10 = p01 / l00
  const l20 = p02 / l00
  const l11 = Math.sqrt(Math.max(p11 - l10 * l10, eps))
  const l21 = (p12 - l20 * l10) / l11
  const l22 = Math.sqrt(Math.max(p22 - l20 * l20 - l21 * l21, eps))
  // Column-major L, times k.
  return new Float32Array([
    k * l00, k * l10, k * l20,
    0, k * l11, k * l21,
    0, 0, k * l22,
  ])
}
