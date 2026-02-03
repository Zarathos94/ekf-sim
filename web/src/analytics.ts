//! The analytics view: the filter's internals made numeric and legible. It reads the richer
//! `Session.analytics()` payload — the nominal state, the ground truth, the 15-vector estimation
//! error, the full 15×15 error covariance, the NEES, and each sensor's latest innovation and NIS —
//! and lays it out as a live state/uncertainty table, scrolling line charts, a covariance-
//! correlation heatmap, and the measurement-update recursion with a per-sensor NIS gate. Pure DOM
//! and 2D canvas, no framework.

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

// --- analytics payload layout (see eskf-wasm Session::analytics) ---
const NOM_P = 0
const NOM_V = 3
const NOM_Q = 6
const NOM_AB = 10
const NOM_GB = 13
const ERR = 26
const COV = 41 // 15×15 row-major
const NEES_POS = 266
const NEES_FULL = 267
const SENSORS = 269 // 9 × [dim, nis, innov, accepted]

const N = 15
const cov = (a: Float32Array, r: number, c: number) => a[COV + r * N + c]!
const sigma = (a: Float32Array, i: number) => Math.sqrt(Math.max(0, cov(a, i, i)))

const SENSOR_NAMES = [
  'GPS position',
  'Barometer',
  'Magnetometer',
  'LiDAR altimeter',
  'UWB range',
  'Optical flow',
  'GPS velocity',
  'Doppler (DVL)',
  'Attitude fix',
]

// χ²(0.99) gate thresholds by measurement dimension — matches the Rust core's nis_gate_threshold.
const GATE: Record<number, number> = { 1: 6.635, 2: 9.21, 3: 11.345, 4: 13.277 }

// scope-trace palette (mirrors the CSS custom properties)
const CY = '#46cde6'
const MN = '#57e3a1'
const AM = '#f4b860'
const CO = '#ff6b78'
const VI = '#ac8dff'

interface Block {
  name: string
  off: number // error-state / covariance offset
  unit: string
  scale: number // multiply stored value for display (rad→deg for orientation)
  estimate: (a: Float32Array) => [number, number, number]
}

const RAD = 180 / Math.PI

const BLOCKS: Block[] = [
  { name: 'Position  δp', off: 0, unit: 'm', scale: 1, estimate: (a) => vec(a, NOM_P) },
  { name: 'Velocity  δv', off: 3, unit: 'm/s', scale: 1, estimate: (a) => vec(a, NOM_V) },
  { name: 'Orientation  δθ', off: 6, unit: '°', scale: RAD, estimate: (a) => eulerDeg(a, NOM_Q) },
  { name: 'Accel bias  δa_b', off: 9, unit: 'm/s²', scale: 1, estimate: (a) => vec(a, NOM_AB) },
  { name: 'Gyro bias  δω_b', off: 12, unit: 'rad/s', scale: 1, estimate: (a) => vec(a, NOM_GB) },
]

function vec(a: Float32Array, off: number): [number, number, number] {
  return [a[off]!, a[off + 1]!, a[off + 2]!]
}

function eulerDeg(a: Float32Array, off: number): [number, number, number] {
  const w = a[off]!
  const x = a[off + 1]!
  const y = a[off + 2]!
  const z = a[off + 3]!
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))))
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return [roll * RAD, pitch * RAD, yaw * RAD]
}

const f = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—')

export class Analytics {
  readonly root: HTMLElement
  private readonly cells: { est: HTMLElement; sig: HTMLElement; err: HTMLElement }[] = []
  private readonly traceEl: HTMLElement
  private readonly sensorRows: {
    dim: HTMLElement
    innov: HTMLElement
    nis: HTMLElement
    gate: HTMLElement
    verdict: HTMLElement
  }[] = []
  private readonly heat: Heatmap
  private readonly errChart: LineChart
  private readonly neesChart: LineChart
  private readonly rateChart: LineChart
  private readonly sigChart: LineChart
  private readonly abChart: LineChart
  private readonly gbChart: LineChart
  private readonly nisChart: LineChart
  private readonly trChart: LineChart
  private readonly grid: HTMLElement

  constructor() {
    this.root = el('div', 'analytics')

    // Header with the dimensionality of the problem, stated up front.
    const head = el('div', 'an-head')
    head.append(el('div', 'an-title', 'Filter internals — live'))
    const dims = el('div', 'an-dims')
    for (const [k, v] of [
      ['nominal state', '16'],
      ['error state', '15'],
      ['covariance P', '15 × 15'],
      ['IMU rate', '200 Hz'],
      ['gate', 'χ²₀.₉₉'],
    ] as const) {
      const chip = el('span', 'an-chip')
      chip.append(el('b', undefined, v), document.createTextNode(' ' + k))
      dims.append(chip)
    }
    head.append(dims)
    head.append(el('div', 'an-hint', '⠿ drag a card by its handle to rearrange'))
    this.root.append(head)

    const grid = el('div', 'an-grid')
    this.grid = grid
    this.root.append(grid)

    const cards: { key: string; el: HTMLElement }[] = []

    // --- State & uncertainty table (full width) ---
    const stateCard = card('state', 'State · uncertainty · error   (x̂ = truth ⊞ error)', true)
    const table = el('table', 'an-table')
    const thead = el('tr')
    for (const h of ['Block', 'dim', 'estimate  x̂', '1σ = √diag(P)', 'error  e = x̂ ⊟ x']) {
      thead.append(el('th', undefined, h))
    }
    table.append(thead)
    for (const b of BLOCKS) {
      const tr = el('tr')
      tr.append(el('td', 'an-bname', b.name), el('td', 'an-dim', '3'))
      const est = el('td', 'an-num')
      const sig = el('td', 'an-num an-sig')
      const err = el('td', 'an-num an-err')
      tr.append(est, sig, err)
      table.append(tr)
      this.cells.push({ est, sig, err })
    }
    stateCard.body.append(table)
    stateCard.body.append(el('div', 'an-note', 'Units per row; orientation as roll/pitch/yaw, its error in degrees.'))
    cards.push({ key: 'state', el: stateCard.card })

    // --- Line charts ---
    const errCard = card('poserr', 'Position error components   (m)')
    this.errChart = new LineChart(['eₓ', 'e_y', 'e_z'], [CY, MN, CO], { symmetric: true })
    errCard.body.append(this.errChart.canvas, this.errChart.legend)
    cards.push({ key: 'poserr', el: errCard.card })

    const sigCard = card('possig', 'Position uncertainty  1σ = √diag(P)   (m)')
    this.sigChart = new LineChart(['σₓ', 'σ_y', 'σ_z'], [CY, MN, CO], { ymin: 0 })
    sigCard.body.append(this.sigChart.canvas, this.sigChart.legend)
    cards.push({ key: 'possig', el: sigCard.card })

    const neesCard = card('nees', 'Consistency · NEES ÷ dof   (hover at 1)')
    this.neesChart = new LineChart(['position ÷ 3', 'full ÷ 15'], [CY, CO], {
      expected: 1,
      band: [0.55, 1.45],
      ymin: 0,
      ymax: 2.4,
    })
    neesCard.body.append(this.neesChart.canvas, this.neesChart.legend)
    cards.push({ key: 'nees', el: neesCard.card })

    const nisCard = card('nis', 'Innovation consistency · max NIS vs gate')
    this.nisChart = new LineChart(['max NIS', 'gate χ²₃'], [CY, CO], { ymin: 0, ymax: 13 })
    nisCard.body.append(this.nisChart.canvas, this.nisChart.legend)
    cards.push({ key: 'nis', el: nisCard.card })

    const rateCard = card('velatt', 'Velocity & attitude error')
    this.rateChart = new LineChart(['‖v‖ err (m/s)', 'att err (°)'], [AM, VI], { ymin: 0 })
    rateCard.body.append(this.rateChart.canvas, this.rateChart.legend)
    cards.push({ key: 'velatt', el: rateCard.card })

    const trCard = card('trace', 'Total uncertainty · tr(P)')
    this.trChart = new LineChart(['tr(P)'], [AM], { ymin: 0 })
    trCard.body.append(this.trChart.canvas, this.trChart.legend)
    cards.push({ key: 'trace', el: trCard.card })

    const abCard = card('abias', 'Accelerometer-bias estimate   (m/s²)')
    this.abChart = new LineChart(['b_ax', 'b_ay', 'b_az'], [CY, MN, CO], { symmetric: true })
    abCard.body.append(this.abChart.canvas, this.abChart.legend)
    cards.push({ key: 'abias', el: abCard.card })

    const gbCard = card('gbias', 'Gyroscope-bias estimate   (rad/s)')
    this.gbChart = new LineChart(['b_gx', 'b_gy', 'b_gz'], [CY, MN, CO], { symmetric: true })
    gbCard.body.append(this.gbChart.canvas, this.gbChart.legend)
    cards.push({ key: 'gbias', el: gbCard.card })

    // --- Covariance correlation heatmap ---
    const heatCard = card('heat', 'Error-covariance correlation  ρ(i,j)   (15 × 15)')
    this.heat = new Heatmap()
    heatCard.body.append(this.heat.canvas, this.heat.legend())
    cards.push({ key: 'heat', el: heatCard.card })

    // --- Measurement update recursion + per-sensor NIS (full width) ---
    const updCard = card('update', 'Measurement update — every correction', true)
    const rec = el('div', 'an-rec')
    rec.innerHTML = [
      '<div>y = z − h(x̂)<span>innovation, m×1</span></div>',
      '<div>S = H P Hᵀ + R<span>m×m</span></div>',
      '<div>K = P Hᵀ S⁻¹<span>gain, 15×m</span></div>',
      '<div>x̂ ← x̂ ⊞ K y<span>inject</span></div>',
      '<div>P⁺ = (I−KH)P(I−KH)ᵀ + KRKᵀ<span>Joseph, 15×15</span></div>',
      '<div>ν = yᵀS⁻¹y ≤ χ²₀.₉₉<span>NIS gate</span></div>',
    ].join('')
    updCard.body.append(rec)
    this.traceEl = el('div', 'an-trace', 'tr(P) = —')
    updCard.body.append(this.traceEl)

    const stable = el('table', 'an-table an-sensors')
    const sh = el('tr')
    for (const h of ['Sensor', 'm', '‖innovation‖', 'NIS', 'gate', '']) sh.append(el('th', undefined, h))
    stable.append(sh)
    for (const name of SENSOR_NAMES) {
      const tr = el('tr')
      const dim = el('td', 'an-num')
      const innov = el('td', 'an-num')
      const nis = el('td', 'an-num')
      const gate = el('td', 'an-num')
      const verdict = el('td', 'an-verdict')
      tr.append(el('td', 'an-sname', name), dim, innov, nis, gate, verdict)
      stable.append(tr)
      this.sensorRows.push({ dim, innov, nis, gate, verdict })
    }
    updCard.body.append(stable)
    cards.push({ key: 'update', el: updCard.card })

    this.mount(cards)
  }

  private charts(): LineChart[] {
    return [
      this.errChart,
      this.sigChart,
      this.neesChart,
      this.nisChart,
      this.rateChart,
      this.trChart,
      this.abChart,
      this.gbChart,
    ]
  }

  clear() {
    for (const c of this.charts()) c.clear()
  }

  // --- card ordering: honour a saved arrangement, then let the user drag to rearrange ---

  private mount(cards: { key: string; el: HTMLElement }[]) {
    const byKey = new Map(cards.map((c) => [c.key, c.el]))
    const saved = this.loadOrder()
    const order = [
      ...saved.filter((k) => byKey.has(k)),
      ...cards.filter((c) => !saved.includes(c.key)).map((c) => c.key),
    ]
    for (const k of order) this.grid.append(byKey.get(k)!)
    this.wireDnd()
  }

  private wireDnd() {
    const grid = this.grid
    let dragEl: HTMLElement | null = null
    grid.querySelectorAll<HTMLElement>('.an-card').forEach((card) => {
      card.addEventListener('dragstart', (e) => {
        dragEl = card
        card.classList.add('dragging')
        ;(e as DragEvent).dataTransfer!.effectAllowed = 'move'
      })
      card.addEventListener('dragend', () => {
        card.classList.remove('dragging')
        dragEl = null
        this.persistOrder()
      })
    })
    grid.addEventListener('dragover', (e) => {
      e.preventDefault()
      if (!dragEl) return
      const de = e as DragEvent
      const target = this.dropTarget(de.clientX, de.clientY, dragEl)
      if (target === dragEl) return
      grid.insertBefore(dragEl, target)
    })
  }

  /** The card the dragged tile should be inserted before (null → append), by nearest centre. */
  private dropTarget(x: number, y: number, dragEl: HTMLElement): HTMLElement | null {
    const cards = [...this.grid.querySelectorAll<HTMLElement>('.an-card')].filter((c) => c !== dragEl)
    let best: HTMLElement | null = null
    let bestDist = Infinity
    let before = true
    for (const c of cards) {
      const r = c.getBoundingClientRect()
      const cx = r.left + r.width / 2
      const cy = r.top + r.height / 2
      const d = Math.hypot(x - cx, y - cy)
      if (d < bestDist) {
        bestDist = d
        best = c
        before = y < cy - 8 || (y <= r.bottom && x < cx)
      }
    }
    if (!best) return null
    return before ? best : (best.nextElementSibling as HTMLElement | null)
  }

  private loadOrder(): string[] {
    try {
      return JSON.parse(localStorage.getItem('eskf.analytics.order') ?? '[]') as string[]
    } catch {
      return []
    }
  }

  private persistOrder() {
    const keys = [...this.grid.querySelectorAll<HTMLElement>('.an-card')].map((c) => c.dataset.key ?? '')
    try {
      localStorage.setItem('eskf.analytics.order', JSON.stringify(keys))
    } catch {
      /* storage unavailable — ordering just won't persist */
    }
  }

  /** Feed one analytics payload; `push` advances the scrolling charts only when running. */
  update(a: Float32Array, push: boolean) {
    for (let i = 0; i < BLOCKS.length; i++) {
      const b = BLOCKS[i]!
      const c = this.cells[i]!
      const est = b.estimate(a)
      const d = b.unit === 'rad/s' ? 4 : b.unit === '°' ? 2 : 3
      c.est.textContent = est.map((v) => f(v, d)).join('  ')
      c.sig.textContent = [0, 1, 2].map((k) => f(sigma(a, b.off + k) * b.scale, d)).join('  ')
      c.err.textContent = [0, 1, 2].map((k) => f(a[ERR + b.off + k]! * b.scale, d)).join('  ')
    }

    // total uncertainty tr(P), and the largest NIS among sensors that fired — both plotted below.
    let trace = 0
    for (let i = 0; i < N; i++) trace += cov(a, i, i)
    let maxNis = 0
    for (let i = 0; i < SENSOR_NAMES.length; i++) {
      if (a[SENSORS + i * 4]! > 0) maxNis = Math.max(maxNis, a[SENSORS + i * 4 + 1]!)
    }

    if (push) {
      this.errChart.push([a[ERR]!, a[ERR + 1]!, a[ERR + 2]!]) // δp per axis
      this.sigChart.push([sigma(a, 0), sigma(a, 1), sigma(a, 2)]) // √diag(P) for p
      this.neesChart.push([a[NEES_POS]! / 3, a[NEES_FULL]! / 15])
      this.nisChart.push([maxNis, GATE[3]!])
      this.rateChart.push([
        Math.hypot(a[ERR + 3]!, a[ERR + 4]!, a[ERR + 5]!),
        Math.hypot(a[ERR + 6]!, a[ERR + 7]!, a[ERR + 8]!) * RAD,
      ])
      this.trChart.push([trace])
      this.abChart.push([a[NOM_AB]!, a[NOM_AB + 1]!, a[NOM_AB + 2]!])
      this.gbChart.push([a[NOM_GB]!, a[NOM_GB + 1]!, a[NOM_GB + 2]!])
    }
    for (const c of this.charts()) c.draw()

    this.heat.draw(a)

    this.traceEl.textContent = `tr(P) = ${trace.toExponential(2)}   ·   position NEES ${f(a[NEES_POS]!, 2)} (exp 3)   ·   full-state NEES ${f(a[NEES_FULL]!, 2)} (exp 15)`

    for (let i = 0; i < SENSOR_NAMES.length; i++) {
      const base = SENSORS + i * 4
      const dim = a[base]!
      const r = this.sensorRows[i]!
      if (dim === 0) {
        r.dim.textContent = '—'
        r.innov.textContent = '—'
        r.nis.textContent = '—'
        r.gate.textContent = '—'
        r.verdict.textContent = ''
        r.verdict.className = 'an-verdict'
        continue
      }
      const nis = a[base + 1]!
      const innov = a[base + 2]!
      const accepted = a[base + 3]! > 0.5
      r.dim.textContent = String(Math.round(dim))
      r.innov.textContent = f(innov, 3)
      r.nis.textContent = f(nis, 2)
      r.gate.textContent = f(GATE[Math.round(dim)] ?? 0, 2)
      r.verdict.textContent = accepted ? 'accept' : 'reject'
      r.verdict.className = 'an-verdict ' + (accepted ? 'ok' : 'bad')
    }
  }
}

function card(key: string, title: string, wide = false): { card: HTMLElement; body: HTMLElement } {
  const c = el('section', wide ? 'an-card wide' : 'an-card')
  c.dataset.key = key
  c.draggable = true
  const t = el('div', 'an-card-title')
  const grip = el('span', 'an-grip', '⠿')
  grip.title = 'drag to rearrange'
  t.append(grip, el('span', undefined, title))
  const body = el('div', 'an-card-body')
  c.append(t, body)
  return { card: c, body }
}

/** A compact multi-series scrolling line chart on a 2D canvas. */
class LineChart {
  readonly canvas: HTMLCanvasElement
  readonly legend: HTMLElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly data: number[][]
  private readonly max = 180
  private readonly h = 120

  constructor(
    labels: string[],
    private readonly colors: string[],
    private readonly opts: {
      symmetric?: boolean
      expected?: number
      band?: [number, number]
      ymin?: number
      ymax?: number
    } = {},
  ) {
    this.canvas = el('canvas', 'an-chart')
    const ctx = this.canvas.getContext('2d')
    if (!ctx) throw new Error('2D canvas unavailable')
    this.ctx = ctx
    this.data = labels.map(() => [])
    this.legend = el('div', 'an-legend')
    for (let i = 0; i < labels.length; i++) {
      const item = el('span', 'an-leg')
      const sw = el('span', 'an-sw')
      sw.style.background = colors[i]!
      item.append(sw, document.createTextNode(labels[i]!))
      this.legend.append(item)
    }
  }

  clear() {
    for (const s of this.data) s.length = 0
  }

  push(vals: number[]) {
    for (let i = 0; i < this.data.length; i++) {
      this.data[i]!.push(vals[i]!)
      if (this.data[i]!.length > this.max) this.data[i]!.shift()
    }
  }

  draw() {
    const ctx = this.ctx
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = this.canvas.clientWidth || 320
    const h = this.h
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr)
      this.canvas.height = Math.round(h * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#0d1320'
    ctx.fillRect(0, 0, w, h)

    const n = this.data[0]!.length
    // y-range
    let lo = this.opts.ymin ?? Infinity
    let hi = this.opts.ymax ?? -Infinity
    if (this.opts.ymin === undefined || this.opts.ymax === undefined) {
      for (const s of this.data) for (const v of s) {
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
      }
      if (this.opts.band) {
        lo = Math.min(lo, this.opts.band[0])
        hi = Math.max(hi, this.opts.band[1])
      }
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        lo = -1
        hi = 1
      }
      if (this.opts.symmetric) {
        const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6)
        lo = -m
        hi = m
      }
      const pad = (hi - lo) * 0.12 || 0.1
      lo -= pad
      hi += pad
    }
    const pl = 40
    const y = (v: number) => h - 4 - ((v - lo) / (hi - lo || 1)) * (h - 8)
    const x = (i: number) => pl + (i / (this.max - 1)) * (w - pl - 4)

    // gridlines + y labels
    ctx.font = '10px ui-monospace, monospace'
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'right'
    for (let g = 0; g <= 2; g++) {
      const v = lo + (g / 2) * (hi - lo)
      const yy = y(v)
      ctx.strokeStyle = 'rgba(120,140,170,0.14)'
      ctx.beginPath()
      ctx.moveTo(pl, yy)
      ctx.lineTo(w - 4, yy)
      ctx.stroke()
      ctx.fillStyle = '#7a8699'
      ctx.fillText(fmtTick(v), pl - 4, yy)
    }

    // acceptance band
    if (this.opts.band) {
      ctx.fillStyle = 'rgba(115,191,255,0.10)'
      ctx.fillRect(pl, y(this.opts.band[1]), w - pl - 4, y(this.opts.band[0]) - y(this.opts.band[1]))
    }
    // expected line
    if (this.opts.expected !== undefined) {
      ctx.strokeStyle = 'rgba(230,235,245,0.5)'
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(pl, y(this.opts.expected))
      ctx.lineTo(w - 4, y(this.opts.expected))
      ctx.stroke()
      ctx.setLineDash([])
    }
    // zero line for symmetric charts
    if (this.opts.symmetric) {
      ctx.strokeStyle = 'rgba(230,235,245,0.35)'
      ctx.beginPath()
      ctx.moveTo(pl, y(0))
      ctx.lineTo(w - 4, y(0))
      ctx.stroke()
    }

    if (n < 2) return
    for (let s = 0; s < this.data.length; s++) {
      ctx.strokeStyle = this.colors[s]!
      ctx.lineWidth = 1.4
      ctx.beginPath()
      const series = this.data[s]!
      for (let i = 0; i < series.length; i++) {
        const px = x(i + (this.max - series.length))
        const py = y(series[i]!)
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.stroke()
    }
  }
}

function fmtTick(v: number): string {
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(0)
  return v.toFixed(a < 1 ? 2 : a < 10 ? 1 : 0)
}

/** The 15×15 correlation matrix as a diverging heatmap, with the five state blocks outlined. */
class Heatmap {
  readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly cell = 15
  private readonly pad = 26

  constructor() {
    this.canvas = el('canvas', 'an-heat')
    const ctx = this.canvas.getContext('2d')
    if (!ctx) throw new Error('2D canvas unavailable')
    this.ctx = ctx
  }

  legend(): HTMLElement {
    const wrap = el('div', 'an-heatleg')
    wrap.append(
      swatch('#3b6fd6', '−1'),
      swatch('#12203a', '0'),
      swatch('#d64b52', '+1'),
      el('span', 'an-note', '  blocks: p · v · θ · a_b · ω_b'),
    )
    return wrap
  }

  draw(a: Float32Array) {
    const ctx = this.ctx
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const size = this.pad + N * this.cell + 2
    if (this.canvas.width !== Math.round(size * dpr)) {
      this.canvas.width = Math.round(size * dpr)
      this.canvas.height = Math.round(size * dpr)
      this.canvas.style.width = size + 'px'
      this.canvas.style.height = size + 'px'
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size, size)

    const sig: number[] = []
    for (let i = 0; i < N; i++) sig.push(sigma(a, i))
    const p0 = this.pad
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const denom = sig[r]! * sig[c]!
        const rho = denom > 1e-20 ? Math.max(-1, Math.min(1, cov(a, r, c) / denom)) : 0
        ctx.fillStyle = diverging(rho)
        ctx.fillRect(p0 + c * this.cell, p0 + r * this.cell, this.cell - 0.5, this.cell - 0.5)
      }
    }
    // block outlines at 0,3,6,9,12,15
    ctx.strokeStyle = 'rgba(230,235,245,0.55)'
    ctx.lineWidth = 1
    for (const b of [0, 3, 6, 9, 12, 15]) {
      ctx.beginPath()
      ctx.moveTo(p0 + b * this.cell, p0)
      ctx.lineTo(p0 + b * this.cell, p0 + N * this.cell)
      ctx.moveTo(p0, p0 + b * this.cell)
      ctx.lineTo(p0 + N * this.cell, p0 + b * this.cell)
      ctx.stroke()
    }
    // block labels
    ctx.fillStyle = '#7a8699'
    ctx.font = '10px ui-monospace, monospace'
    ctx.textBaseline = 'middle'
    const labels = ['p', 'v', 'θ', 'a', 'ω']
    for (let i = 0; i < 5; i++) {
      const mid = p0 + (i * 3 + 1.5) * this.cell
      ctx.textAlign = 'center'
      ctx.fillText(labels[i]!, mid, 12)
      ctx.textAlign = 'right'
      ctx.fillText(labels[i]!, this.pad - 6, mid)
    }
  }
}

function diverging(t: number): string {
  // t in [-1,1] → blue (neg) · dark (0) · red (pos)
  const neg = [59, 111, 214]
  const zero = [18, 32, 58]
  const pos = [214, 75, 82]
  const m = t < 0 ? mix(zero, neg, -t) : mix(zero, pos, t)
  return `rgb(${m[0]},${m[1]},${m[2]})`
}
function mix(a: number[], b: number[], t: number): number[] {
  return [0, 1, 2].map((i) => Math.round(a[i]! + (b[i]! - a[i]!) * t))
}
function swatch(color: string, label: string): HTMLElement {
  const s = el('span', 'an-leg')
  const sw = el('span', 'an-sw')
  sw.style.background = color
  s.append(sw, document.createTextNode(label))
  return s
}
