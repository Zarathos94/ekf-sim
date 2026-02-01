//! The control surface: the 3D stage, the sensor toggles and noise sliders that drive the
//! simulator, the error/consistency readouts, and a live plot of the position error against its
//! own 3σ envelope — the consistency claim, made visible. Pure DOM, no framework.

import type { Session } from './wasm/eskf_wasm.js'
import { Analytics } from './analytics.js'

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (cls) node.className = cls
  if (text !== undefined) node.textContent = text
  return node
}

interface SensorDef {
  label: string
  on: boolean
  pulse: number
  color: string
  set: (s: Session, v: boolean) => void
}

// Order matches the snapshot's pulse block [36..45].
const SENSORS: SensorDef[] = [
  { label: 'GPS position', on: true, pulse: 0, color: '#73bfff', set: (s, v) => s.set_gps_enabled(v) },
  { label: 'Barometer', on: true, pulse: 1, color: '#8ad', set: (s, v) => s.set_baro_enabled(v) },
  { label: 'Magnetometer', on: true, pulse: 2, color: '#c9a', set: (s, v) => s.set_mag_enabled(v) },
  { label: 'LiDAR altimeter', on: false, pulse: 3, color: '#ffbe59', set: (s, v) => s.set_lidar_enabled(v) },
  { label: 'UWB radio ranging', on: false, pulse: 4, color: '#c78bff', set: (s, v) => s.set_uwb_enabled(v) },
  { label: 'Optical flow', on: false, pulse: 5, color: '#5be68d', set: (s, v) => s.set_flow_enabled(v) },
  { label: 'GPS velocity', on: false, pulse: 6, color: '#5bb0e6', set: (s, v) => s.set_gps_vel_enabled(v) },
  { label: 'Doppler velocity (DVL)', on: false, pulse: 7, color: '#e69a5b', set: (s, v) => s.set_dvl_enabled(v) },
  { label: 'Attitude fix (vision)', on: false, pulse: 8, color: '#e65b9a', set: (s, v) => s.set_att_enabled(v) },
]

interface Preset {
  name: string
  // Enabled state in SENSORS order: gps, baro, mag, lidar, uwb, flow, gps_vel, dvl, att.
  on: boolean[]
}

const PRESETS: Preset[] = [
  { name: 'Aided INS (GPS + baro + mag)', on: [true, true, true, false, false, false, false, false, false] },
  { name: 'Full fusion (all 10)', on: [true, true, true, true, true, true, true, true, true] },
  { name: 'GPS-denied (UWB ranging)', on: [false, true, true, false, true, false, false, false, false] },
  { name: 'Indoor (UWB + LiDAR + flow)', on: [false, true, true, true, true, true, false, false, false] },
  { name: 'Vision-aided (attitude + DVL)', on: [false, true, true, false, false, false, false, true, true] },
  { name: 'Dead reckoning (IMU only)', on: [false, false, false, false, false, false, false, false, false] },
]

export class Ui {
  readonly canvas: HTMLCanvasElement
  readonly analytics = new Analytics()
  activeView: '3d' | 'analytics' = '3d'
  paused = false
  onReset: () => void = () => {}
  onExaggerate: (n: number) => void = () => {}

  private readonly rPos: HTMLElement
  private readonly rAtt: HTMLElement
  private readonly rNees: HTMLElement
  private readonly rNeesDot: HTMLElement
  private readonly rTime: HTMLElement
  private readonly dots: HTMLElement[] = []
  private readonly sensorInputs: HTMLInputElement[] = []
  private readonly plot: ConsistencyPlot

  constructor(root: HTMLElement, session: Session) {
    root.classList.add('eskf-root')

    // --- Stage: the 3D canvas with a floating legend and metric readouts. ---
    const stage = el('div', 'stage')
    this.canvas = el('canvas', 'scene')
    stage.append(this.canvas)

    const hud = el('div', 'hud')
    hud.append(
      legend('#73bfff', 'Estimate'),
      legend('#5be68d', 'Ground truth'),
      legend('#ff5a66', 'Position error'),
      legend('#ffbe59', '95% uncertainty'),
      legend('#c78bff', 'UWB beacon'),
    )
    stage.append(hud)

    const metrics = el('div', 'metrics')
    const posM = metric('Position error')
    const attM = metric('Attitude error')
    const neesM = metric('Position NEES')
    const timeM = metric('Flight time')
    this.rNeesDot = el('span', 'dot')
    neesM.value.prepend(this.rNeesDot)
    metrics.append(posM.wrap, attM.wrap, neesM.wrap, timeM.wrap)
    stage.append(metrics)

    // View switcher: the 3D stage, or the numeric analytics panel over the same area.
    this.analytics.root.style.display = 'none'
    stage.append(this.analytics.root)
    const tabs = el('div', 'tabs')
    const t3d = el('button', 'tab active', '3D view')
    const tan = el('button', 'tab', 'Analytics')
    const setView = (v: '3d' | 'analytics') => {
      this.activeView = v
      const on3d = v === '3d'
      t3d.classList.toggle('active', on3d)
      tan.classList.toggle('active', !on3d)
      this.analytics.root.style.display = on3d ? 'none' : 'block'
      hud.style.display = on3d ? '' : 'none'
      metrics.style.display = on3d ? '' : 'none'
    }
    t3d.addEventListener('click', () => setView('3d'))
    tan.addEventListener('click', () => setView('analytics'))
    tabs.append(t3d, tan)
    stage.append(tabs)

    this.rPos = posM.value
    this.rAtt = attM.value
    this.rNees = neesM.value
    this.rTime = timeM.value

    // --- Rail: a masthead over hairline-separated control sections. ---
    const rail = el('div', 'rail')
    const mast = el('div', 'mast')
    const mrow = el('div', 'mast-row')
    mrow.append(
      el('span', 'mast-logo', '◈'),
      el('span', 'mast-word', 'ESKF'),
      el('span', 'mast-tag', 'sensor-fusion lab'),
    )
    mast.append(
      mrow,
      el(
        'div',
        'mast-desc',
        'A quaternion error-state Kalman filter fusing an IMU with GPS, baro, mag, LiDAR, UWB, optical flow, DVL and attitude fixes.',
      ),
    )
    const body = el('div', 'rail-body')

    body.append(section('01', 'Display'))
    body.append(slider('Error exaggeration', 1, 20, 4, 1, '×', (v) => this.onExaggerate(v)))

    body.append(section('02', 'Scenario'))
    const presetSel = el('select', 'preset')
    presetSel.append(el('option', undefined, '— choose a preset —'))
    for (const p of PRESETS) presetSel.append(el('option', undefined, p.name))
    presetSel.addEventListener('change', () => {
      const p = PRESETS[presetSel.selectedIndex - 1]
      if (!p) return
      for (let i = 0; i < SENSORS.length; i++) {
        this.sensorInputs[i]!.checked = p.on[i]!
        SENSORS[i]!.set(session, p.on[i]!)
      }
    })
    body.append(presetSel)

    body.append(section('03', 'Sensors'))
    for (const def of SENSORS) {
      def.set(session, def.on)
      const row = el('label', 'sw')
      const input = el('input')
      input.type = 'checkbox'
      input.checked = def.on
      input.addEventListener('change', () => def.set(session, input.checked))
      this.sensorInputs.push(input)
      const track = el('span', 'sw-track')
      track.append(el('span', 'sw-knob'))
      const dot = el('span', 'sw-dot')
      dot.style.background = def.color
      dot.style.opacity = '0.15'
      this.dots.push(dot)
      row.append(input, track, el('span', 'sw-name', def.label), dot)
      body.append(row)
    }

    body.append(section('04', 'IMU noise & bias drift'))
    body.append(
      slider('Accel noise', 0.0, 1.0, 0.06, 0.005, 'm/s²', (v) => session.set_accel_noise(v)),
      slider('Gyro noise', 0.0, 0.08, 0.004, 0.001, 'rad/s', (v) => session.set_gyro_noise(v)),
      slider('Accel bias drift', 0.0, 0.03, 0.002, 0.0005, 'm/s²/√s', (v) => session.set_accel_bias_walk(v)),
      slider('Gyro bias drift', 0.0, 0.004, 0.0002, 0.0001, 'rad/s/√s', (v) => session.set_gyro_bias_walk(v)),
    )

    body.append(section('05', 'Aiding-sensor noise'))
    body.append(
      slider('GPS noise', 0.1, 8.0, 0.8, 0.1, 'm', (v) => session.set_gps_noise(v)),
      slider('LiDAR noise', 0.02, 2.0, 0.15, 0.02, 'm', (v) => session.set_lidar_noise(v)),
      slider('UWB noise', 0.05, 3.0, 0.35, 0.05, 'm', (v) => session.set_uwb_noise(v)),
      slider('Optical-flow noise', 0.02, 1.0, 0.15, 0.02, 'm/s', (v) => session.set_flow_noise(v)),
      slider('GPS-velocity noise', 0.02, 1.0, 0.1, 0.02, 'm/s', (v) => session.set_gps_vel_noise(v)),
      slider('DVL noise', 0.01, 0.5, 0.05, 0.01, 'm/s', (v) => session.set_dvl_noise(v)),
      slider('Attitude-fix noise', 0.002, 0.1, 0.01, 0.002, 'rad', (v) => session.set_att_noise(v)),
    )

    body.append(section('06', 'Consistency · error vs 3σ'))
    this.plot = new ConsistencyPlot()
    body.append(this.plot.canvas)
    body.append(
      el('p', 'plotcap', 'The error (white) should stay inside the ±3σ envelope (amber) the filter reports.'),
    )

    body.append(section('07', 'Run'))
    const buttons = el('div', 'buttons')
    const pauseBtn = el('button', 'btn', 'Pause')
    pauseBtn.addEventListener('click', () => {
      this.paused = !this.paused
      pauseBtn.textContent = this.paused ? 'Resume' : 'Pause'
      pauseBtn.classList.toggle('active', this.paused)
    })
    const resetBtn = el('button', 'btn', 'New flight')
    resetBtn.addEventListener('click', () => {
      session.reset()
      this.plot.clear()
      this.analytics.clear()
      this.onReset()
    })
    buttons.append(pauseBtn, resetBtn)
    body.append(buttons)

    body.append(
      el(
        'p',
        'hint',
        'Drag to orbit, scroll to zoom. The red spear is the position error, exaggerated ×4 by default — set it to ×1 for true scale. Turn every aiding sensor off to watch the estimate dead-reckon and drift; drop GPS and turn UWB on to stay localised with no GPS. The Analytics tab shows the covariance, NEES and per-sensor NIS in numbers.',
      ),
    )

    rail.append(mast, body)
    root.append(stage, rail)
  }

  updateReadouts(s: Float32Array) {
    const posErr = s[26]!
    const attErr = s[27]!
    const nees = s[28]!
    const t = s[29]!
    const sigma3 = 3 * s[45]!
    const instErr = s[46]!

    this.rPos.lastChild!.textContent = `${posErr.toFixed(2)} m`
    this.rAtt.textContent = `${attErr.toFixed(2)}°`
    this.rNees.lastChild!.textContent = ` ${nees.toFixed(2)}`
    let cls = 'dot ok'
    let hint = 'consistent'
    if (nees > 6.0) {
      cls = 'dot bad'
      hint = 'overconfident'
    } else if (nees < 1.2) {
      cls = 'dot warn'
      hint = 'conservative'
    }
    this.rNeesDot.className = cls
    this.rNeesDot.title = hint
    this.rTime.textContent = `${t.toFixed(1)} s`

    // Sensor activity dots, driven by the decaying pulses — they light and glow on each fix.
    for (let i = 0; i < this.dots.length; i++) {
      const p = s[36 + i]!
      const dot = this.dots[i]!
      dot.style.opacity = (0.18 + 0.82 * p).toFixed(2)
      dot.style.boxShadow = p > 0.15 ? `0 0 ${(7 * p).toFixed(1)}px ${SENSORS[i]!.color}` : 'none'
    }

    if (!this.paused) this.plot.push(instErr, sigma3)
    this.plot.draw()
  }
}

/** A scrolling plot of the position error against its 3σ envelope. */
class ConsistencyPlot {
  readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly err: number[] = []
  private readonly sig: number[] = []
  private readonly max = 200

  constructor() {
    this.canvas = el('canvas', 'plot')
    const ctx = this.canvas.getContext('2d')
    if (!ctx) throw new Error('2D canvas unavailable')
    this.ctx = ctx
  }

  clear() {
    this.err.length = 0
    this.sig.length = 0
  }

  push(err: number, sigma3: number) {
    this.err.push(err)
    this.sig.push(sigma3)
    if (this.err.length > this.max) {
      this.err.shift()
      this.sig.shift()
    }
  }

  draw() {
    const ctx = this.ctx
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = this.canvas.clientWidth || 264
    const h = 96
    if (this.canvas.width !== Math.round(w * dpr)) {
      this.canvas.width = Math.round(w * dpr)
      this.canvas.height = Math.round(h * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#0c0e13'
    ctx.fillRect(0, 0, w, h)

    const n = this.err.length
    if (n < 2) return
    let top = 0.5
    for (let i = 0; i < n; i++) top = Math.max(top, this.sig[i]!, this.err[i]!)
    top *= 1.15
    const x = (i: number) => (i / (this.max - 1)) * w
    const y = (v: number) => h - (v / top) * (h - 6) - 3

    // 3σ envelope (filled).
    ctx.beginPath()
    ctx.moveTo(x(0), y(this.sig[0]!))
    for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(this.sig[i]!))
    ctx.lineTo(x(n - 1), h)
    ctx.lineTo(x(0), h)
    ctx.closePath()
    ctx.fillStyle = 'rgba(244,184,96,0.15)'
    ctx.fill()
    ctx.strokeStyle = 'rgba(244,184,96,0.85)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(x(0), y(this.sig[0]!))
    for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(this.sig[i]!))
    ctx.stroke()

    // Error line.
    ctx.strokeStyle = 'rgba(235,240,250,0.95)'
    ctx.lineWidth = 1.25
    ctx.beginPath()
    ctx.moveTo(x(0), y(this.err[0]!))
    for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(this.err[i]!))
    ctx.stroke()
  }
}

function legend(color: string, label: string): HTMLElement {
  const wrap = el('span', 'legend')
  const swatch = el('span', 'swatch')
  swatch.style.background = color
  wrap.append(swatch, document.createTextNode(label))
  return wrap
}

function metric(label: string): { wrap: HTMLElement; value: HTMLElement } {
  const wrap = el('div', 'metric')
  wrap.append(el('span', 'k', label))
  const v = el('span', 'v')
  v.append(document.createTextNode('—'))
  wrap.append(v)
  return { wrap, value: v }
}

/** A numbered, hairline section divider (the number is drawn by CSS from data-n). */
function section(n: string, text: string): HTMLElement {
  const s = el('div', 'section', text)
  s.dataset.n = n
  return s
}

/**
 * A machined control: the name, a `[− value +]` stepper (no ugly native spin buttons — the value
 * is editable and the −/+ keys step by the given increment), a unit, and a range slider, all synced.
 */
function slider(
  label: string,
  min: number,
  max: number,
  value: number,
  step: number,
  unit: string,
  onInput: (v: number) => void,
): HTMLElement {
  const decimals = (String(step).split('.')[1] ?? '').length
  const clamp = (v: number) => Number(Math.min(max, Math.max(min, v)).toFixed(decimals))

  const wrap = el('div', 'ctl')
  const top = el('div', 'ctl-top')
  const name = el('span', 'ctl-name', label)
  name.title = label

  const stepper = el('div', 'stepper')
  const minus = el('button', 'step minus', '−')
  minus.type = 'button'
  const num = el('input', 'stepval')
  num.type = 'number'
  num.min = String(min)
  num.max = String(max)
  num.step = String(step)
  num.value = String(value)
  const plus = el('button', 'step plus', '+')
  plus.type = 'button'
  stepper.append(minus, num, plus)

  top.append(name, stepper, el('span', 'ctl-unit', unit))

  const range = el('input')
  range.type = 'range'
  range.min = String(min)
  range.max = String(max)
  range.step = String(step)
  range.value = String(value)

  // range/stepper are the source of truth; `keepText` leaves the number field alone while typing.
  const apply = (v: number, keepText = false) => {
    const c = clamp(v)
    range.value = String(c)
    if (!keepText) num.value = String(c)
    onInput(c)
  }
  range.addEventListener('input', () => apply(Number(range.value)))
  minus.addEventListener('click', () => apply(Number(range.value) - step))
  plus.addEventListener('click', () => apply(Number(range.value) + step))
  num.addEventListener('input', () => {
    const raw = Number(num.value)
    if (!Number.isNaN(raw)) apply(raw, true)
  })
  num.addEventListener('blur', () => apply(Number(num.value) || min))

  wrap.append(top, range)
  return wrap
}
