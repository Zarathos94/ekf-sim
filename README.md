# ekf-sim — a quaternion error-state EKF, proven consistent

A sensor-fusion playground: an **error-state Kalman filter** (ESKF) fuses a simulated strapdown
IMU with GPS, a barometer, a magnetometer, a downward **LiDAR** altimeter, **UWB radio ranging**
to fixed beacons, and **optical flow** to estimate a quadrotor's trajectory. The estimate, the
ground truth (as a lit quadrotor and a faint ghost) and the live **covariance ellipsoid** are
rendered in 3D. Toggle any sensor, turn up the noise or the bias drift, and watch the filter
degrade and recover — the uncertainty ellipsoid swelling and shrinking in real units, and a live
plot showing the error stay inside its own 3σ envelope. Drop GPS and the ellipsoid balloons; turn
on UWB ranging and it snaps back, localised with no GPS at all.

The point of the project is that the quaternion ESKF is done *right*, and that this is **proven,
not asserted**. The gate is statistical consistency: across many Monte-Carlo flights the filter's
error, weighed by its own reported covariance (the NEES), must match the state dimension. A filter
that merely tracks well but reports the wrong uncertainty fails this — which is exactly how a
subtly wrong Jacobian or a mis-scaled noise term is caught.

```
$ cargo run -p eskf-cli -- check

Monte-Carlo consistency check — 40 flights of 30 s, all sensors on

  RMSE   position 0.083 m   velocity 0.043 m/s   attitude 0.230°

  position    NEES mean  2.898   expected 3   95% band [2.289, 3.805]   ok
  full state  NEES mean 16.514   expected 15   95% band [13.350, 16.744]   ok

CONSISTENT — the filter's covariance matches its error. Gate PASSED.
```

A full technical write-up — the error-state derivation, the nine measurement Jacobians, the
observability analysis, the NEES/NIS consistency methodology, and the results (with figures generated
directly by the filter) — is in **[`paper/ekf-sim.pdf`](paper/ekf-sim.pdf)**
([LaTeX source](paper/ekf-sim.tex)).

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
- **Updates**, each with its exact error-state Jacobian (finite-difference-checked):
  - **GPS** — 3-D position.
  - **Barometer** — altitude.
  - **Magnetometer** — the full field vector in the body frame (heading as a vector measurement,
    not a yaw shortcut).
  - **LiDAR altimeter** — the slant range down to flat ground, `p_z / R₂₂`, so it senses both
    height and tilt.
  - **UWB radio ranging** — scalar range to each of four fixed beacons; position-only, so it
    localises the vehicle with no GPS at all.
  - **Optical flow** — the body-frame horizontal velocity, observing velocity and attitude.
  - **GPS Doppler velocity** — a direct world-frame velocity fix.
  - **Doppler velocity (DVL / radar)** — the full body-frame velocity.
  - **Attitude fix** — a direct orientation measurement (star tracker / vision), the standard
    multiplicative attitude update.

  Joseph-form covariance update throughout, with the orientation covariance-reset Jacobian on
  injection. A recorded reference dataset ([`data/reference-flight.csv`](data/reference-flight.csv))
  and a `replay` command make the results independently reproducible.

See [`docs/eskf.md`](docs/eskf.md) for the equations, worked and cited.

## Layout

```
crates/
  eskf/       the filter, the quaternion algebra, the simulator, and the NEES metric — no deps,
              #![forbid(unsafe_code)], 24 native tests
  eskf-cli/   the correctness gate: `check` (Monte-Carlo NEES) and `scenarios` (RMSE by failure)
  eskf-wasm/  wasm-bindgen surface: steps sim + filter live for the browser
web/          Vite + TypeScript, a hand-written WebGL2 scene (no framework)
```

## Build

```
cargo test --workspace                 # the core, natively (27 tests)
cargo run -p eskf-cli -- check         # the Monte-Carlo NEES gate, pooled + over time
cargo run -p eskf-cli -- scenarios     # RMSE across 9 sensor/failure scenarios
cargo run -p eskf-cli -- record 15 data/reference-flight.csv   # record a dataset
cargo run -p eskf-cli -- replay data/reference-flight.csv      # replay + score it
cargo run -p eskf-cli -- plotdata paper/figdata               # regenerate the paper's figure data

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
