"""
telemetry.py
------------
Real-time parameter telemetry that stays OUT of the terminal.

A single record() call in the control loop feeds two outputs:
  1. JSONL log     — every snapshot appended to a timestamped file for
                     later replay/analysis (pandas, matplotlib, etc.)
  2. Live dashboard — a local web page that plots rolling time-series,
                     so you glance at a browser tab instead of watching
                     numbers scroll past.

The dashboard uses Server-Sent Events (SSE) over Python's stdlib HTTP
server: NO extra dependencies, works fully offline. SSE is one-way
(server → browser), which is all telemetry needs.

Usage:
    tel = Telemetry()
    tel.start()                      # prints the dashboard URL
    tel.record(features, params)     # call every control cycle
    tel.stop()

Open the printed http://localhost:8765 in any browser while the app runs.
"""

import json
import queue
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import paths


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that doesn't spew tracebacks when a browser drops an
    SSE connection (EventSource reconnects, favicon probes, tab close)."""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError,
                            ConnectionAbortedError)):
            return   # benign client disconnect — ignore
        super().handle_error(request, client_address)


class Telemetry:
    def __init__(self, log_path: str | None = None, port: int = 8765):
        if log_path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = paths.telemetry_path(f"telemetry-{stamp}.jsonl")
        self._log_path = log_path
        self._port = port
        self._log_file = None
        self._clients: list[queue.Queue] = []   # one queue per SSE browser tab
        self._clients_lock = threading.Lock()
        self._server = None
        self._t0 = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        self._t0 = time.monotonic()
        self._log_file = open(self._log_path, "a", buffering=1)  # line-buffered
        self._server = _QuietServer(
            ("127.0.0.1", self._port), self._make_handler()
        )
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[telemetry] dashboard → http://localhost:{self._port}")
        print(f"[telemetry] logging    → {self._log_path}")

    def record(self, features, params, out_level=0.0, out_bright=0.0,
               out_spectrum=None):
        """Append one 20Hz snapshot (input features + output audio) to the
        log and push it to live dashboards."""
        snap = {
            "kind":   "tick",
            "t":      round(time.monotonic() - self._t0, 3),
            "energy": round(features.energy, 4),
            "pitch":  round(features.pitch, 1),
            "rate":   round(features.speech_rate, 4),
            "bright": round(features.brightness, 4),
            "silent": features.is_silent,
            "blend":  round(params.prompt_blend, 4),
            "chaos":  round(params.chaos, 4),
            "drums":  params.drums_on,
            "out_level":  round(out_level, 4),    # RMS of audio leaving speakers
            "out_bright": round(out_bright, 4),   # spectral centroid of that audio
        }
        if out_spectrum:
            snap["spec"] = out_spectrum          # log-spaced FFT bands (0..1)
        self._emit(json.dumps(snap))

    def event(self, kind: str, **fields):
        """Push a discrete event (scene change, hold, heard, render) — drawn as
        a marker + log line on the dashboard, separate from the 20Hz stream."""
        obj = {"kind": kind, "t": round(time.monotonic() - self._t0, 3), **fields}
        self._emit(json.dumps(obj))

    def _emit(self, line: str):
        """Log a line and fan it out to all connected dashboards."""
        if self._log_file:
            self._log_file.write(line + "\n")
        with self._clients_lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(line)
                except queue.Full:
                    dead.append(q)   # slow/stuck client — drop it
            for q in dead:
                self._clients.remove(q)

    def stop(self):
        if self._server:
            self._server.shutdown()
        if self._log_file:
            self._log_file.close()

    # ── Internal: SSE client registry ─────────────────────────────────

    def _register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=600)   # ~30s at 20Hz before dropping
        with self._clients_lock:
            self._clients.append(q)
        return q

    def _unregister(self, q: queue.Queue):
        with self._clients_lock:
            if q in self._clients:
                self._clients.remove(q)

    # ── Internal: HTTP handler factory ────────────────────────────────

    def _make_handler(self):
        telemetry = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence per-request stderr logging — keep terminal calm

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    self._serve_page()
                elif self.path == "/stream":
                    self._serve_stream()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_page(self):
                body = _DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                # Allow the control panel (different port) to read this stream so
                # its equalizer can react to the live MRT2 audio level.
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                q = telemetry._register()
                try:
                    while True:
                        try:
                            line = q.get(timeout=15)
                        except queue.Empty:
                            # heartbeat comment keeps the connection alive
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser tab closed
                finally:
                    telemetry._unregister(q)

        return Handler


# ══════════════════════════════════════════════════════════════════════
#  Dashboard page — self-contained, no CDN, hand-rolled canvas charts.
#  Works offline. Renders a rolling window of the last ~15s at 20 Hz.
# ══════════════════════════════════════════════════════════════════════

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Score the Story — Telemetry</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #0d1117; color: #e6edf3;
    font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid #21262d;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .3px; }
  #conn { font-size: 12px; padding: 2px 8px; border-radius: 10px; }
  .up   { background:#1a3a2a; color:#3fb950; }
  .down { background:#3a1a1a; color:#f85149; }
  #viewbtn { font: inherit; font-size:12px; cursor:pointer; padding:3px 12px;
             border-radius:10px; background:#161b22; color:#c9d1d9;
             border:1px solid #30363d; }
  #viewbtn:hover { background:#21262d; border-color:#484f58; }
  .pills { display: flex; gap: 10px; margin-left: auto; }
  .pill {
    font-size: 12px; padding: 3px 10px; border-radius: 10px;
    background: #161b22; border: 1px solid #21262d; color: #8b949e;
  }
  .pill.on { color: #fff; }
  .pill.silent.on { background:#30363d; border-color:#484f58; color:#c9d1d9; }
  .pill.drums.on  { background:#3a2a1a; border-color:#9e6a3a; color:#ffa657; }
  main { padding: 16px 20px; }
  canvas { width: 100%; height: 560px; display: block; background:#010409;
           border:1px solid #21262d; border-radius: 8px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin:12px 2px 20px; }
  .legend span { display:flex; align-items:center; gap:6px; font-size:13px; }
  .sw { width:11px; height:11px; border-radius:2px; display:inline-block; }
  .val { color:#fff; font-variant-numeric: tabular-nums; min-width:46px; text-align:right; }
  .pitch { display:flex; align-items:center; gap:10px; font-size:13px; color:#8b949e; }
  .pitch b { color:#fff; font-variant-numeric: tabular-nums; }
  .panels { display:grid; grid-template-columns: 1fr 1.4fr; gap:16px; margin-top:18px; }
  .panels section { background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:12px 14px; }
  .panels h2 { font-size:12px; margin:0 0 10px; color:#8b949e; font-weight:600;
               text-transform:uppercase; letter-spacing:.5px; }
  .lat { font-size:14px; color:#8b949e; }
  .lat b { color:#fff; font-variant-numeric: tabular-nums; }
  .log { font-size:12px; line-height:1.7; max-height:260px; overflow-y:auto; }
  .log div { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .lg-h { color:#58a6ff; }   /* heard      */
  .lg-s { color:#3fb950; }   /* scene      */
  .lg-x { color:#6e7681; }   /* hold       */
  .lg-r { color:#d29922; }   /* render     */
  @media (max-width: 720px) { .panels { grid-template-columns: 1fr; } }

  /* ── Pipeline strip: Voice → Whisper → Claude → MRT2 → Out ───────────── */
  .pipeline { display:flex; align-items:center; gap:6px; padding:14px 20px;
              border-bottom:1px solid #21262d; flex-wrap:wrap; }
  .pnode { display:flex; flex-direction:column; align-items:center; gap:3px;
           padding:8px 14px; border-radius:10px; background:#0d1117;
           border:1px solid #21262d; min-width:96px; transition:all .12s; }
  .pnode .ic { font-size:18px; }
  .pnode .nm { font-size:11px; color:#8b949e; letter-spacing:.3px; }
  .pnode.on { border-color:#3fb950; background:#0f2417;
              box-shadow:0 0 14px #3fb95055; }
  .pnode.on .nm { color:#3fb950; }
  .pnode.mrt2 { min-width:120px; }
  .pnode.mrt2.on { border-color:#bc8cff; background:#1a1228;
                   box-shadow:0 0 20px #bc8cff66; }
  .pnode.mrt2.on .nm { color:#d2b3ff; }
  .parrow { color:#30363d; font-size:16px; }

  /* ── MRT2 engine hero panel ─────────────────────────────────────────── */
  .engine { margin:16px 20px 0; background:linear-gradient(180deg,#15101f,#0d0a14);
            border:1px solid #3a2a55; border-radius:12px; padding:16px 18px; }
  .engine h2 { margin:0 0 12px; font-size:13px; color:#d2b3ff; font-weight:700;
               letter-spacing:.5px; display:flex; align-items:center; gap:8px; }
  .gendot { width:9px; height:9px; border-radius:50%; background:#6e7681; }
  .gendot.live { background:#bc8cff; box-shadow:0 0 10px #bc8cff;
                 animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  .eng-grid { display:grid; grid-template-columns: 1.4fr .6fr; gap:18px; align-items:center; }
  .poles { font-size:13px; }
  .pole { display:flex; align-items:center; gap:8px; margin:2px 0; }
  .pole .tag { color:#e6edf3; }
  .pole.a .dot{ color:#f85149 } .pole.b .dot{ color:#3fb950 }
  .blendbar { position:relative; height:8px; border-radius:4px; margin:10px 0 4px;
              background:linear-gradient(90deg,#3a1a1a,#1a3a1a); }
  .blenddot { position:absolute; top:-3px; width:14px; height:14px; border-radius:50%;
              background:#fff; box-shadow:0 0 8px #fff8; transform:translateX(-50%);
              transition:left .12s; }
  .eng-meta { display:flex; gap:10px; flex-wrap:wrap; font-size:12px; margin-top:8px; }
  .chip { padding:3px 9px; border-radius:9px; background:#1c1530; border:1px solid #3a2a55;
          color:#c9d1d9; }
  .chip b { color:#fff; }
  .vu { height:14px; border-radius:7px; background:#1c1530; overflow:hidden; }
  .vu .fill { height:100%; width:0%; background:linear-gradient(90deg,#bc8cff,#d2b3ff);
              transition:width .1s; }
  .eng-out { font-size:11px; color:#8b949e; margin-top:6px; }
</style>
</head>
<body>
<header>
  <h1>🎙 Score the Story — Telemetry</h1>
  <span id="conn" class="down">connecting…</span>
  <button id="viewbtn">Live 15 s</button>
  <div class="pills">
    <span id="p-silent" class="pill silent">silent</span>
    <span id="p-drums"  class="pill drums">drums</span>
  </div>
</header>

<!-- Pipeline: each stage lights up as it fires. MRT2 is the engine in the middle. -->
<div class="pipeline">
  <div class="pnode" id="pn-voice"><span class="ic">🎙</span><span class="nm">YOUR VOICE</span></div>
  <span class="parrow">▸</span>
  <div class="pnode" id="pn-whisper"><span class="ic">📝</span><span class="nm">WHISPER</span></div>
  <span class="parrow">▸</span>
  <div class="pnode" id="pn-claude"><span class="ic">🧠</span><span class="nm">CLAUDE</span></div>
  <span class="parrow">▸</span>
  <div class="pnode mrt2" id="pn-mrt2"><span class="ic">♪</span><span class="nm">MAGENTA RT2</span></div>
  <span class="parrow">▸</span>
  <div class="pnode" id="pn-out"><span class="ic">🔊</span><span class="nm">SPEAKERS</span></div>
</div>

<!-- MRT2 engine: what it's being fed (style poles + blend + key + drums) and generating. -->
<section class="engine">
  <h2><span class="gendot" id="gendot"></span> MAGENTA REALTIME 2 — live generative engine</h2>
  <div class="eng-grid">
    <div>
      <div class="poles">
        <div class="pole a"><span class="dot">●</span> A: <span class="tag" id="eng-a">—</span></div>
        <div class="pole b"><span class="dot">●</span> B: <span class="tag" id="eng-b">—</span></div>
      </div>
      <div class="blendbar"><div class="blenddot" id="eng-blenddot" style="left:50%"></div></div>
      <div class="eng-meta">
        <span class="chip">key <b id="eng-key">—</b></span>
        <span class="chip">drums <b id="eng-drums">off</b></span>
        <span class="chip">generating <b id="eng-gen">—</b> ms/chunk</span>
      </div>
    </div>
    <div>
      <div class="vu"><div class="fill" id="eng-vu"></div></div>
      <div class="eng-out">MRT2 audio output level</div>
    </div>
  </div>
</section>

<main>
  <canvas id="chart"></canvas>
  <div class="pitch">pitch <b id="pitch">— Hz</b> · solid = voice in · dashed = audio out · │ = scene change · <span id="axis">window: whole session</span></div>
  <div class="panels">
    <section>
      <h2>Last scene change — latency</h2>
      <div id="lat" class="lat">waiting for first scene change…</div>
    </section>
    <section>
      <h2>Events</h2>
      <div id="log" class="log"></div>
    </section>
  </div>
</main>

<script>
const WINDOW = 300;             // points in the "live" view (~15s at 20Hz)
const MAXPOINTS = 36000;        // full-session cap (~30 min at 20Hz)
let viewMode = 'full';          // 'full' = whole session, 'live' = last 15s
let T0 = 0;                     // global tick index of data[0] (after capping)
const SERIES = {
  energy:     { color:'#f85149', label:'energy',     dash:false },
  bright:     { color:'#d29922', label:'bright',     dash:false },
  rate:       { color:'#58a6ff', label:'rate',       dash:false },
  blend:      { color:'#3fb950', label:'blend',      dash:false },
  chaos:      { color:'#bc8cff', label:'chaos',      dash:false },
  out_level:  { color:'#ffffff', label:'out:level',  dash:true  },
  out_bright: { color:'#ff7b72', label:'out:bright', dash:true  },
};
for (const k in SERIES) SERIES[k].data = [];

let T = 0;            // global tick counter, for placing markers
let markers = [];     // {T, label} scene-change markers
let lastScene = null; // {whisper, claude, render} latency of last change

// Build legend with live value readouts
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr;
  canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resize);
resize();

// Group the series into labelled bands so related signals sit together.
const GROUPS = [
  { title: 'voice in',   keys: ['energy', 'bright', 'rate'] },
  { title: 'mapped',     keys: ['blend', 'chaos'] },
  { title: 'audio out',  keys: ['out_level', 'out_bright'] },
];
const ROWS = GROUPS.reduce((n, g) => n + g.keys.length, 0);

function draw() {
  const r = canvas.getBoundingClientRect();
  const W = r.width, H = r.height;
  const LABEL_W = 92;                 // left gutter for names + values
  const rowH = H / ROWS;
  ctx.clearRect(0, 0, W, H);

  // x-axis mapping: full session (all points) or live window (last WINDOW).
  const N = SERIES.energy.data.length;
  const plotW = W - LABEL_W;
  let i0, span;
  if (viewMode === 'live') {
    const count = Math.min(N, WINDOW);
    i0 = N - count; span = Math.max(1, count - 1);
  } else {
    i0 = 0; span = Math.max(1, N - 1);
  }
  const xOf = (i) => LABEL_W + ((i - i0) / span) * plotW;

  // Scene-change markers span every row (shared time axis).
  ctx.font = '10px monospace';
  for (const m of markers) {
    const di = m.T - T0;               // data index of this marker
    if (di < i0 || di > N - 1) continue;
    const x = xOf(di);
    ctx.strokeStyle = '#3fb95055'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3fb950';
    ctx.save(); ctx.translate(x + 3, 4); ctx.rotate(Math.PI / 2);
    ctx.fillText(m.label.slice(0, 26), 0, 0); ctx.restore();
  }

  let row = 0;
  for (const group of GROUPS) {
    for (let gi = 0; gi < group.keys.length; gi++) {
      const k = group.keys[gi];
      const s = SERIES[k];
      const top = row * rowH, bot = (row + 1) * rowH;
      const ipad = 8;
      const y0 = bot - ipad;          // value 0 (bottom of band)
      const y1 = top + ipad;          // value 1 (top of band)

      // band separator + faint mid gridline
      ctx.setLineDash([]);
      ctx.strokeStyle = '#1b2129'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, bot); ctx.lineTo(W, bot); ctx.stroke();
      ctx.strokeStyle = '#11161c';
      const ymid = y1 + (y0 - y1) * 0.5;
      ctx.beginPath(); ctx.moveTo(LABEL_W, ymid); ctx.lineTo(W, ymid); ctx.stroke();

      // group title on the first row of each group
      if (gi === 0) {
        ctx.fillStyle = '#484f58'; ctx.font = '9px monospace';
        ctx.fillText(group.title.toUpperCase(), 6, top + 11);
      }
      // series label + live value
      const cur = s.data.length ? s.data[s.data.length - 1] : 0;
      ctx.fillStyle = s.color; ctx.font = '11px monospace';
      ctx.fillText(s.label, 6, top + (gi === 0 ? 24 : 16));
      ctx.fillStyle = '#fff';
      ctx.fillText(cur.toFixed(2), LABEL_W - 34, top + (gi === 0 ? 24 : 16));

      // the trace (only the visible slice, per the current view mode)
      const d = s.data;
      if (N - i0 >= 2) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.dash ? 1.4 : 1.6;
        ctx.setLineDash(s.dash ? [5, 3] : []);
        ctx.beginPath();
        for (let i = i0; i < N; i++) {
          const x = xOf(i);
          const v = Math.max(0, Math.min(1, d[i]));
          const y = y0 - (y0 - y1) * v;
          (i === i0) ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      row++;
    }
  }
  ctx.setLineDash([]);
}

const conn = document.getElementById('conn');
const es = new EventSource('/stream');
es.onopen  = () => { conn.textContent = 'live';        conn.className = 'up'; };
es.onerror = () => { conn.textContent = 'reconnecting…'; conn.className = 'down'; };

// View toggle: full session ⇄ live 15s window.
const viewbtn = document.getElementById('viewbtn');
const axisLabel = document.getElementById('axis');
function applyView() {
  viewbtn.textContent = (viewMode === 'full') ? 'Live 15 s' : 'Full session';
  if (axisLabel) axisLabel.textContent =
      (viewMode === 'full') ? 'window: whole session' : 'window: last 15 s';
}
viewbtn.addEventListener('click', () => {
  viewMode = (viewMode === 'full') ? 'live' : 'full';
  applyView();
});
applyView();
function logLine(html) {
  const log = document.getElementById('log');
  const div = document.createElement('div');
  div.innerHTML = html;
  log.prepend(div);
  while (log.children.length > 14) log.removeChild(log.lastChild);
}

function updateLat() {
  if (!lastScene) return;
  const w = lastScene.whisper ?? '—', c = lastScene.claude ?? '—',
        r = lastScene.render ?? '…';
  const tot = (lastScene.whisper||0) + (lastScene.claude||0) + (lastScene.render||0);
  document.getElementById('lat').innerHTML =
    `whisper <b>${w}</b>ms · claude <b>${c}</b>ms · render <b>${r}</b>ms` +
    `  →  total <b>${(tot/1000).toFixed(1)}</b>s`;
}

// ── Pipeline strip + MRT2 engine panel state ──────────────────────────────
let lastHeardMs = -9999, lastClaudeMs = -9999, lastSceneFlash = -9999, lastGen = null;
const $ = (id) => document.getElementById(id);
function lit(id, on) { $(id).classList.toggle('on', on); }

function updateEngine(d) {           // called on every tick
  const now = performance.now();
  const out = d.out_level ?? 0, energy = d.energy ?? 0;
  // Pipeline: each stage lights when it's active.
  lit('pn-voice',   energy > 0.02);
  lit('pn-whisper', now - lastHeardMs  < 700);
  lit('pn-claude',  now - lastClaudeMs < 900);
  lit('pn-mrt2',    out > 0.015 || now - lastSceneFlash < 600);   // generating
  lit('pn-out',     out > 0.015);
  // MRT2 engine: live blend dot, drums, output meter, generating pulse.
  $('eng-blenddot').style.left = ((d.blend ?? 0.5) * 100).toFixed(0) + '%';
  $('eng-drums').textContent = d.drums ? 'ON' : 'off';
  $('eng-vu').style.width = Math.min(100, (out / 0.4) * 100).toFixed(0) + '%';
  $('gendot').classList.toggle('live', out > 0.015);
  if (lastGen != null) $('eng-gen').textContent = lastGen;
}

es.onmessage = (e) => {
  const d = JSON.parse(e.data);

  if (d.kind === 'tick' || d.kind === undefined) {
    let shifted = false;
    for (const k in SERIES) {
      SERIES[k].data.push(d[k] ?? 0);
      if (SERIES[k].data.length > MAXPOINTS) { SERIES[k].data.shift(); shifted = true; }
    }
    if (shifted) T0++;            // oldest point dropped — advance its tick index
    document.getElementById('pitch').textContent =
        (d.pitch > 0 ? d.pitch.toFixed(0) : '—') + ' Hz';
    document.getElementById('p-silent').classList.toggle('on', d.silent);
    document.getElementById('p-drums').classList.toggle('on', d.drums);
    updateEngine(d);
    T++;
    markers = markers.filter(m => m.T >= T0);   // keep markers still in range
    return;
  }

  // Discrete events
  if (d.kind === 'heard') {
    lastHeardMs = performance.now();
    logLine(`<span class="lg-h">🎙 ${d.text}</span>`);
  } else if (d.kind === 'scene') {
    markers.push({ T, label: d.a });
    lastScene = { whisper: d.whisper, claude: d.claude, render: null };
    lastClaudeMs = lastSceneFlash = performance.now();
    $('eng-a').textContent = d.a;
    $('eng-b').textContent = d.b;
    $('eng-key').textContent = d.key || '—';
    updateLat();
    logLine(`<span class="lg-s">▶ SCENE — A: ${d.a} | B: ${d.b}</span>`);
  } else if (d.kind === 'hold') {
    lastClaudeMs = performance.now();
    logLine(`<span class="lg-x">• same scene — holding</span>`);
  } else if (d.kind === 'render') {
    if (lastScene) { lastScene.render = d.ms; updateLat(); }
    lastGen = d.chunk; lastSceneFlash = performance.now();
    const warn = d.underruns > 0 ? ` ⚠ underruns ${d.underruns}` : '';
    logLine(`<span class="lg-r">♪ rendered ${d.ms}ms (${d.chunk}ms/chunk)${warn}</span>`);
  }
};

// Decouple render rate from data rate — redraw on animation frames
(function loop(){ draw(); requestAnimationFrame(loop); })();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# Quick standalone test: streams synthetic data to the dashboard.
#   python3 telemetry.py   → open http://localhost:8765
# ------------------------------------------------------------------
if __name__ == "__main__":
    import math
    from types import SimpleNamespace

    tel = Telemetry(log_path="telemetry-test.jsonl")
    tel.start()
    print("Streaming synthetic data — open the dashboard. Ctrl+C to stop.")
    try:
        i = 0
        while True:
            p = i / 100.0
            feats = SimpleNamespace(
                energy=0.5 + 0.45 * math.sin(p),
                pitch=120 + 60 * math.sin(p * 0.7),
                speech_rate=0.5 + 0.4 * math.sin(p * 1.3),
                brightness=0.5 + 0.45 * math.sin(p * 0.5),
                is_silent=(math.sin(p * 0.3) < -0.7),
            )
            params = SimpleNamespace(
                prompt_blend=0.5 + 0.45 * math.sin(p * 0.5),
                chaos=0.5 + 0.4 * math.sin(p),
                drums_on=(math.sin(p) > 0.3),
            )
            out_level  = 0.4 + 0.3 * math.sin(p * 1.1)
            out_bright = 0.4 + 0.3 * math.sin(p * 0.6)
            tel.record(feats, params, out_level, out_bright)

            # Fire synthetic events so the markers / log / latency panel show.
            if i % 120 == 0:
                tel.event("heard", text="the creature erupted from the depths")
                tel.event("scene", a="low pulsing bass drone",
                          b="driving urgent strings", whisper=82, claude=1463)
                tel.event("render", ms=4735, chunk=779, underruns=0)
            elif i % 40 == 0:
                tel.event("hold")
            i += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        tel.stop()
        print("\nStopped.")
