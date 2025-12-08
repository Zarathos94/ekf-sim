# ekf-sim — a quaternion error-state EKF, proven consistent

A sensor-fusion playground: an **error-state Kalman filter** (ESKF) fuses a simulated strapdown
IMU with GPS, a barometer and a magnetometer to estimate a vehicle's trajectory, and renders the
estimate, the ground truth and the live **covariance ellipsoid** in 3D. Turn up the sensor noise,
the bias drift, or drop GPS, and watch the filter degrade and recover — the uncertainty ellipsoid
swelling and shrinking in real units as it does.

The point of the project is that the quaternion ESKF is done *right*, and that this is **proven,
not asserted**. The gate is statistical consistency: across many Monte-Carlo flights the filter's
error, weighed by its own reported covariance (the NEES), must match the state dimension. A filter
that merely tracks well but reports the wrong uncertainty fails this — which is exactly how a
subtly wrong Jacobian or a mis-scaled noise term is caught.

```
$ cargo run -p eskf-cli -- check

Monte-Carlo consistency check — 40 flights of 30 s, all sensors on

  RMSE   position 0.431 m   velocity 0.218 m/s   attitude 0.860°

  position    NEES mean  2.704   expected 3   95% band [2.289, 3.805]   ok
  full state  NEES mean 14.794   expected 15   95% band [13.350, 16.744]   ok

CONSISTENT — the filter's covariance matches its error. Gate PASSED.
```

## The filter

Error-state (indirect) Kalman filter in the formulation of Solà, *Quaternion kinematics for the
error-state Kalman filter* (2017), with the local (body-frame) angular error.

- **Nominal state** (16): position, velocity, orientation quaternion, accelerometer bias, gyro
  bias — integrated directly from the IMU.
- **Error state** (15): `[δp, δv, δθ, δa_b, δω_b]`, always reset to zero between steps, so its
  15×15 covariance stays well conditioned and the orientation error is a minimal 3-vector instead
  of an over-parameterised quaternion.
- **Prediction** propagates the error covariance through the block-structured transition; the
  orientation block uses the exact `exp(−[ω]×dt)`.
- **Updates**: GPS (3-D position), barometer (altitude), magnetometer (the full field vector in
  the body frame — heading done as a vector measurement, not a yaw shortcut). Joseph-form
  covariance update, and the orientation covariance-reset Jacobian on injection.

See [`docs/eskf.md`](docs/eskf.md) for the equations, worked and cited.

## Layout

```
crates/
  eskf/       the filter, the quaternion algebra, the simulator, and the NEES metric — no deps,
              #![forbid(unsafe_code)], 20 native tests
  eskf-cli/   the correctness gate: `check` (Monte-Carlo NEES) and `scenarios` (RMSE by failure)
  eskf-wasm/  wasm-bindgen surface: steps sim + filter live for the browser
web/          Vite + TypeScript, a hand-written WebGL2 scene (no framework)
```

## Build

```
cargo test --workspace           # the core, natively
cargo run -p eskf-cli -- check   # the consistency gate
cargo run -p eskf-cli -- scenarios

./scripts/build-wasm.sh          # regenerate web/src/wasm from the Rust
cd web && npm install && npm run dev
```

## Verification

| Layer | How | Criterion |
|---|---|---|
| Linear algebra, quaternions | `cargo test -p eskf` | exp/log round-trips, known rotations, matrix inverse |
| Simulator | `cargo test -p eskf` | analytic velocity/acceleration match finite differences; RNG is standard-normal |
| **Filter** | `cargo run -p eskf-cli -- check` | **position and full-state NEES inside the 95% χ² band** |
| Behaviour | `cargo run -p eskf-cli -- scenarios` | GPS dropout inflates position error while attitude holds |
| Browser | `npm run dev` | estimate tracks truth; the ellipsoid grows on dropout and recovers |

## Licence

MIT — see [LICENSE](LICENSE).
