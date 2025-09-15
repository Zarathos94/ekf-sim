# The error-state Kalman filter, worked

Notation and structure follow Solà, *Quaternion kinematics for the error-state Kalman filter*
(arXiv:1711.02508, 2017). The world frame is East-North-Up-like: `z` is up, gravity is
`g = [0, 0, −9.80665]`. Orientation is a Hamilton unit quaternion `q` (body→world), and its error
is the **local** (body-frame) rotation `δθ` with `q_true = q_nominal ⊗ exp(½ δθ)`.

## Why an error state

A unit quaternion has four components but only three degrees of freedom. Estimating it directly
forces the covariance to describe a constrained 4-vector, which it cannot do without either
drifting off the unit sphere or carrying a singular direction. The error-state filter estimates
instead a minimal 3-vector `δθ` living in the tangent space at the current nominal orientation.
The nominal state holds the full, unconstrained estimate; the error state — reset to zero after
every update — stays small enough that its linearisation is excellent and its covariance is a
clean 3×3 per rotation block.

## State

Nominal `x = (p, v, q, a_b, ω_b)`. Error `δx = (δp, δv, δθ, δa_b, δω_b) ∈ ℝ¹⁵`.

## Prediction (per IMU sample, rate ~200 Hz)

With accelerometer `a_m`, gyro `ω_m`, biases removed `a = a_m − a_b`, `ω = ω_m − ω_b`, and
`R = R(q)`:

**Nominal integration**

```
a_world = R a + g
p ← p + v Δt + ½ a_world Δt²
v ← v + a_world Δt
q ← q ⊗ exp(½ ω Δt)          (renormalised)
```

**Error-state transition** `δx ← F δx`, with `F` block-structured (rows/cols in state order):

```
δp:  I,   IΔt on δv
δv:  I on δv,   −R[a]× Δt on δθ,   −R Δt on δa_b
δθ:  exp(−[ω]× Δt) on δθ,   −I Δt on δω_b
δa_b: I
δω_b: I
```

`[·]×` is the skew-symmetric cross-product matrix. The `δv←δθ` term is the first-order effect of a
body rotation error on the rotated specific force; the `δθ←δθ` term is the exact rotation-integral
`exp(−[ω]×Δt)` (computed via Rodrigues, reusing the quaternion exponential).

**Process noise** `P ← F P Fᵀ + Q`, `Q` block-diagonal. The IMU white noise is a per-sample
standard deviation `σ`, so a noisy reading perturbs the integrated velocity/orientation by `σ Δt`
and injects variance `σ² Δt²` (the only dimensionally consistent choice). The biases are random
walks with rate density `σ_b`, variance `σ_b² Δt`:

```
Q_v  = σ_accel² Δt² I     Q_θ   = σ_gyro² Δt² I
Q_ab = σ_ab² Δt I         Q_ωb  = σ_ωb² Δt I
```

Getting these exponents wrong is invisible in the RMSE and glaring in the NEES — the consistency
gate is what pins them down.

## Updates

For a measurement `z` with model `h(x)`, Jacobian `H = ∂h/∂δx` at the nominal:

```
S = H P Hᵀ + R
K = P Hᵀ S⁻¹
δx = K (z − h(x))                     inject, then reset δx ← 0
P ← (I − K H) P (I − K H)ᵀ + K R Kᵀ   (Joseph form)
```

The Joseph form is used rather than the shorter `(I − KH)P` because it stays symmetric
positive-definite under finite precision.

- **GPS** — `h = p`, `H = [I₃ 0 0 0 0]`.
- **Barometer** — `h = p_z`, `H` selects the up component of `δp`.
- **Magnetometer** — `h = Rᵀ m_ref`, the world reference field measured in the body frame. Its
  Jacobian is `∂(Rᵀ m)/∂δθ = [Rᵀ m]×`, giving heading observability from the full vector, not a
  yaw pseudo-measurement.

## Injection and reset

```
p ← p + δp,   v ← v + δv,   q ← q ⊗ exp(½ δθ),   a_b ← a_b + δa_b,   ω_b ← ω_b + δω_b
P ← G P Gᵀ,   G = diag(I, I, I − ½[δθ]×, I, I)
```

The reset Jacobian `G` accounts for moving the error's linearisation point onto the new nominal
orientation. It is close to the identity, and dropping it — as many implementations do — is one of
the small errors the NEES catches.

## Consistency — the gate

The Normalized Estimation Error Squared is `ε = eᵀ P⁻¹ e`, where `e` is the estimation error
expressed in the same 15-dimensional tangent space the covariance lives in (position and velocity
differences, the body-frame log map for orientation, bias differences). For a consistent filter,
`E[ε] = dim`. Averaged over `N` independent Monte-Carlo flights, `N·ε̄ ∼ χ²(N·dim)`, giving the
two-sided acceptance band the CLI reports. Position (3-DOF) and the full state (15-DOF) are both
checked; a filter that tracks well but reports the wrong uncertainty lands outside the band.
