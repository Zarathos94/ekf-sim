//! The control surface: the 3D stage, the sensor sliders that drive the simulator, and the
//! error/consistency readouts. Pure DOM, no framework.

import type { Session } from './wasm/eskf_wasm.js'

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

interface Readouts {
  pos: HTMLElement
  att: HTMLElement
  nees: HTMLElement
  neesDot: HTMLElement
  gps: HTMLElement
  time: HTMLElement
}

export class Ui {
  readonly canvas: HTMLCanvasElement
  paused = false
  onReset: () => void = () => {}

  private readonly r: Readouts

  constructor(root: HTMLElement, session: Session) {
    root.classList.add('eskf-root')

    const stage = el('div', 'stage')
    this.canvas = el('canvas', 'scene')
    stage.append(this.canvas)

    // Legend + readouts float over the stage.
    const hud = el('div', 'hud')
    hud.append(
      legend('#73bfff', 'Estimate'),
      legend('#5be68d', 'Ground truth'),
      legend('#ffbe59', '95% uncertainty'),
    )
    stage.append(hud)

    const metrics = el('div', 'metrics')
    const posM = metric('Position error', '—')
    const attM = metric('Attitude error', '—')
    const neesM = metric('Position NEES', '—')
    const gpsM = metric('GPS', '—')
    const timeM = metric('Flight time', '—')
    const neesDot = el('span', 'dot')
    neesM.value.prepend(neesDot)
    metrics.append(posM.wrap, attM.wrap, neesM.wrap, gpsM.wrap, timeM.wrap)
    stage.append(metrics)

    this.r = {
      pos: posM.value,
      att: attM.value,
      nees: neesM.value,
      neesDot,
      gps: gpsM.value,
      time: timeM.value,
    }

    // Control rail.
    const rail = el('div', 'rail')
    const title = el('div', 'title')
    title.append(el('h1', undefined, 'Sensor Fusion Playground'), el('p', undefined, 'Error-state EKF · IMU + GPS + baro + magnetometer'))
    rail.append(title)

    rail.append(sectionLabel('Sensors — turn up the noise'))
    rail.append(
      slider('Accel noise', 0.0, 1.0, 0.06, 0.005, 'm/s²', (v) => session.set_accel_noise(v)),
      slider('Gyro noise', 0.0, 0.08, 0.004, 0.001, 'rad/s', (v) => session.set_gyro_noise(v)),
      slider('Accel bias drift', 0.0, 0.03, 0.002, 0.0005, 'm/s²/√s', (v) => session.set_accel_bias_walk(v)),
      slider('Gyro bias drift', 0.0, 0.004, 0.0002, 0.0001, 'rad/s/√s', (v) => session.set_gyro_bias_walk(v)),
      slider('GPS noise', 0.1, 8.0, 0.8, 0.1, 'm', (v) => session.set_gps_noise(v)),
      slider('Baro noise', 0.1, 6.0, 0.6, 0.1, 'm', (v) => session.set_baro_noise(v)),
      slider('Mag noise', 0.0, 0.2, 0.02, 0.005, '', (v) => session.set_mag_noise(v)),
    )

    rail.append(sectionLabel('Failures'))
    const gpsToggle = toggle('GPS dropout', (on) => session.set_gps_dropout(on))
    rail.append(gpsToggle)

    rail.append(sectionLabel('Run'))
    const controls = el('div', 'buttons')
    const pauseBtn = el('button', 'btn', 'Pause')
    pauseBtn.addEventListener('click', () => {
      this.paused = !this.paused
      pauseBtn.textContent = this.paused ? 'Resume' : 'Pause'
      pauseBtn.classList.toggle('active', this.paused)
    })
    const resetBtn = el('button', 'btn', 'New flight')
    resetBtn.addEventListener('click', () => {
      session.reset()
      this.onReset()
    })
    controls.append(pauseBtn, resetBtn)
    rail.append(controls)

    rail.append(
      el(
        'p',
        'hint',
        'Drag to orbit, scroll to zoom. Watch the ellipsoid swell when GPS drops — the filter still knows where it is, and how unsure it has become.',
      ),
    )

    root.append(stage, rail)
  }

  updateReadouts(s: Float32Array) {
    const posErr = s[26]!
    const attErr = s[27]!
    const nees = s[28]!
    const gpsLock = s[29]! > 0.5
    const t = s[30]!

    this.r.pos.lastChild!.textContent = `${posErr.toFixed(2)} m`
    this.r.att.textContent = `${attErr.toFixed(2)}°`
    // NEES ≈ 3 is consistent; well above means overconfident, well below conservative.
    this.r.nees.lastChild!.textContent = ` ${nees.toFixed(2)}`
    let cls = 'dot ok'
    let hint = 'consistent'
    if (nees > 6.0) {
      cls = 'dot bad'
      hint = 'overconfident'
    } else if (nees < 1.2) {
      cls = 'dot warn'
      hint = 'conservative'
    }
    this.r.neesDot.className = cls
    this.r.neesDot.title = hint
    this.r.gps.textContent = gpsLock ? 'locked' : 'DROPOUT'
    this.r.gps.classList.toggle('alert', !gpsLock)
    this.r.time.textContent = `${t.toFixed(1)} s`
  }
}

function legend(color: string, label: string): HTMLElement {
  const wrap = el('span', 'legend')
  const swatch = el('span', 'swatch')
  swatch.style.background = color
  wrap.append(swatch, document.createTextNode(label))
  return wrap
}

function metric(label: string, value: string): { wrap: HTMLElement; value: HTMLElement } {
  const wrap = el('div', 'metric')
  wrap.append(el('span', 'k', label))
  const v = el('span', 'v')
  v.append(document.createTextNode(value))
  wrap.append(v)
  return { wrap, value: v }
}

function sectionLabel(text: string): HTMLElement {
  return el('div', 'section', text)
}

function slider(
  label: string,
  min: number,
  max: number,
  value: number,
  step: number,
  unit: string,
  onInput: (v: number) => void,
): HTMLElement {
  const wrap = el('div', 'slider')
  const head = el('div', 'slabel')
  head.append(el('span', undefined, label))
  const val = el('span', 'sval', fmt(value, unit))
  head.append(val)
  const input = el('input')
  input.type = 'range'
  input.min = String(min)
  input.max = String(max)
  input.step = String(step)
  input.value = String(value)
  input.addEventListener('input', () => {
    const v = Number(input.value)
    val.textContent = fmt(v, unit)
    onInput(v)
  })
  wrap.append(head, input)
  return wrap
}

function toggle(label: string, onChange: (on: boolean) => void): HTMLElement {
  const wrap = el('label', 'toggle')
  const input = el('input')
  input.type = 'checkbox'
  input.addEventListener('change', () => onChange(input.checked))
  wrap.append(input, el('span', undefined, label))
  return wrap
}

function fmt(v: number, unit: string): string {
  const s = v < 0.01 && v > 0 ? v.toExponential(1) : v.toFixed(v < 1 ? 3 : 2)
  return unit ? `${s} ${unit}` : s
}
