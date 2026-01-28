//! Entry point: bring up the WASM session, the WebGL2 scene and the controls, then run the
//! fixed loop — advance the simulator and filter, then draw the newest estimate, trajectory and
//! covariance ellipsoid. Everything is main-thread: the filter is a 15-state EKF at a couple of
//! hundred hertz, which is nothing, so there is no worker to justify.

import './styles.css'
import init, { Session, beacon_positions } from './wasm/eskf_wasm.js'
import wasmUrl from './wasm/eskf_wasm_bg.wasm?url'
import { Scene } from './gl/scene.js'
import { Ui } from './ui.js'

async function main() {
  const root = document.getElementById('app')
  if (!root) throw new Error('missing #app mount')

  const wasm = await init(wasmUrl)
  const memory = wasm.memory

  const session = new Session(BigInt(Date.now()) & 0xffffffffn)
  const ui = new Ui(root, session)
  const scene = new Scene(ui.canvas, beacon_positions())
  ui.onExaggerate = (n) => scene.setExaggeration(n)

  window.addEventListener('resize', () => scene.resize())

  let last = performance.now()
  const frame = (now: number) => {
    const dt = Math.min((now - last) / 1000, 0.05)
    last = now

    if (!ui.paused) session.step(dt)

    // Copy of the scalar snapshot (JS-owned), then fresh views over the trails in wasm memory —
    // recreated every frame so a heap growth never leaves us reading a detached buffer.
    const snapshot = session.snapshot()
    ui.updateReadouts(snapshot) // the rail's consistency plot stays live in either view

    if (ui.activeView === '3d') {
      const estTrail = new Float32Array(memory.buffer, session.est_trail_ptr(), session.est_trail_len())
      const truthTrail = new Float32Array(memory.buffer, session.truth_trail_ptr(), session.truth_trail_len())
      scene.render({ estTrail, truthTrail, snapshot })
    } else {
      ui.analytics.update(session.analytics(), !ui.paused)
    }
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

main().catch((err) => {
  const root = document.getElementById('app')
  if (root) {
    root.innerHTML = `<div style="padding:2rem;color:#e6e9f0;font-family:system-ui">
      <h2>Could not start</h2><p style="color:#9aa4b8">${String(err)}</p></div>`
  }
  // eslint-disable-next-line no-console
  console.error(err)
})
