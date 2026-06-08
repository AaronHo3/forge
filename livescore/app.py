"""
app.py — Score the Story control panel
--------------------------------------
A local web app that makes the whole experience clickable for a non-technical
user, demonstrating BOTH hackathon challenges in one screen:

  • Magenta (MRT2): Start telling → speak → the live generative engine scores
    you in real time. The embedded telemetry dashboard shows it generating.
  • Audiotool: one "Publish to Audiotool" button turns the told story into an
    editable, multiplayer Audiotool project and hands back a shareable link.

No terminal, flags, env vars, or npm for the user — just buttons. Runs entirely
on Python's stdlib HTTP server (same approach as telemetry.py), so no new deps.

Run:
    python3 app.py                 # opens control panel on http://localhost:8000
    python3 app.py --mode midi     # if you drive MRT2 via the AU plugin instead

Setup note (one-time, by whoever installs it): put an Audiotool Personal Access
Token in audiotool_export/.env so Publish can write projects. See that README.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import paths
from engine import ScoreEngine
from telemetry import Telemetry

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(ROOT, "audiotool_export")
STUDIO_RE = re.compile(r"https://beta\.audiotool\.com/studio\?project=[\w-]+")


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


# ── Publish: keepsake → arrangement → Audiotool project (subprocess pipeline) ──
def _latest_keepsake(engine: ScoreEngine) -> str | None:
    if engine.last_keepsake and os.path.exists(engine.last_keepsake):
        return engine.last_keepsake
    files = sorted(glob.glob(os.path.join(paths.KEEPSAKES, "keepsake-*.json")))
    return files[-1] if files else None


def _publish(engine: ScoreEngine) -> dict:
    keepsake = _latest_keepsake(engine)
    if not keepsake:
        return {"error": "No story yet — tell one and press Stop first."}

    log: list[str] = []
    try:
        # 1) Arrange the keepsake into an editable score (Claude melody if key set).
        arrange = subprocess.run(
            [sys.executable, "audiotool_arranger.py", keepsake],
            cwd=ROOT, capture_output=True, text=True, timeout=180)
        log.append(arrange.stdout.strip() or arrange.stderr.strip())
        if arrange.returncode != 0:
            return {"error": f"Arrange failed: {arrange.stderr[:300]}", "log": "\n".join(log)}
        arr_abs = paths.arrangement_for(keepsake)

        # 2) Write it into a real Audiotool project via the Nexus exporter.
        if not os.path.isdir(os.path.join(EXPORT_DIR, "node_modules")):
            return {"error": "audiotool_export deps missing — run `npm install` there once.",
                    "log": "\n".join(log)}
        env = {**os.environ, "ARRANGEMENT": arr_abs}
        export = subprocess.run(
            ["npm", "start", "--silent"], cwd=EXPORT_DIR, env=env,
            capture_output=True, text=True, timeout=180)
        out = (export.stdout or "") + (export.stderr or "")
        log.append(out.strip())
        match = STUDIO_RE.search(out)
        if not match:
            hint = ("No project URL returned. Is AT_PAT set in audiotool_export/.env?"
                    if "AT_PAT" in out or "missing" in out.lower() else "Export produced no URL.")
            return {"error": hint, "log": "\n".join(log)}
        return {"url": match.group(0), "log": "\n".join(log)}
    except subprocess.TimeoutExpired:
        return {"error": "Publish timed out.", "log": "\n".join(log)}
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        return {"error": str(e), "log": "\n".join(log)}


def _prompt_render_keepsake(engine: ScoreEngine) -> None:
    """On quit, offer to render the offline keepsake song from the last telling.
    The live pass is the sketch; this layered offline render is the keepsake."""
    src = engine.last_keepsake
    abs_src = os.path.join(ROOT, src) if src else None
    if not abs_src or not os.path.exists(abs_src):
        return   # no story was told this session — nothing to render
    try:
        ans = input(
            f"\nRender the offline keepsake song from {src}?\n"
            f"  (loads MRT2 and layers per-scene stems — a few minutes) [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  Skipped.")
        return
    if ans not in ("y", "yes"):
        print(f"  Skipped. Render later with:  python3 keepsake.py render {src}")
        return
    print("  Rendering keepsake… progress below.\n")
    # Run in a FRESH process. The live engine already initialised MLX/Metal in
    # THIS process, and a second in-process model load fails with an MLX stream
    # error ("no Stream(gpu, 1) in current thread"). A subprocess gets a clean
    # Metal context — the same pattern Publish uses. Output is NOT captured so
    # the render's progress streams straight to the terminal.
    try:
        proc = subprocess.run([sys.executable, "keepsake.py", "render", src], cwd=ROOT)
        if proc.returncode == 0:
            engine.last_song = paths.song_for(src)
            print(f"\n✓ Keepsake song saved: {engine.last_song}")
        else:
            print(f"  Render failed (exit {proc.returncode}).")
            print(f"  Retry with:  python3 keepsake.py render {src}")
    except Exception as e:  # noqa: BLE001 — report, don't crash the shutdown
        print(f"  Render failed: {e}")
        print(f"  Retry with:  python3 keepsake.py render {src}")


def _make_handler(engine: ScoreEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj: dict, code: int = 200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state":
                self._json(engine.state())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/start":
                # Run off the HTTP thread: model load can take seconds; the UI
                # stays responsive and polls /state for the "starting" phase.
                threading.Thread(target=engine.start, daemon=True).start()
                self._json(engine.state())
            elif self.path == "/stop":
                threading.Thread(target=engine.stop, daemon=True).start()
                self._json(engine.state())
            elif self.path == "/inject":
                ok = engine.inject(self._read_body().get("text", ""))
                self._json({"ok": ok})
            elif self.path == "/mute":
                self._json({"muted": engine.toggle_mute()})
            elif self.path == "/publish":
                self._json(_publish(engine))
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Score the Story — control panel")
    parser.add_argument("--mode", choices=["python", "midi"], default="python",
                        help="python = generate audio directly (default); midi = drive MRT2 AU")
    parser.add_argument("--preset", default="storytelling")
    parser.add_argument("--port", type=int, default=8000, help="control panel port")
    parser.add_argument("--telemetry-port", type=int, default=8765)
    args = parser.parse_args()

    # Persistent telemetry dashboard — started once, lives for the whole app, so
    # the live-engine iframe never breaks across Start/Stop cycles.
    telemetry = Telemetry(port=args.telemetry_port)
    telemetry.start()
    engine = ScoreEngine(mode=args.mode, preset_name=args.preset,
                         telemetry=telemetry, telemetry_port=args.telemetry_port)
    server = _QuietServer(("127.0.0.1", args.port), _make_handler(engine))
    url = f"http://localhost:{args.port}"
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Score the Story — Control Panel                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Open in your browser →  {url}\n")
    print("  Click Start, tell your story, click Stop, then Publish to Audiotool.")
    print("  (Ctrl+C here to quit the server.)\n")
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down…")
        engine.stop()                       # saves the session JSON (the keepsake log)
        server.shutdown()                   # stop serving before the long render
        _prompt_render_keepsake(engine)     # offer the offline .wav render
        telemetry.stop()


# ── Single-page control UI — "Vibrant Music App" art direction ─────────────────
#   Near-black ink, magenta→violet→teal gradient, Space Grotesk, an animated
#   equalizer hero, and glowing gradient pill buttons. Deliberately NOT the
#   default dark/monospace look.
_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Score the Story</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0a0612; --grad:linear-gradient(90deg,#ff2d9b,#7b3ff5,#19d3c5);
    --grad2:linear-gradient(135deg,#ff2d9b,#7b3ff5 55%,#19d3c5);
    --mag:#ff2d9b; --vio:#7b3ff5; --teal:#19d3c5;
    --txt:#f3eefc; --mut:#9a8fb8; --line:rgba(255,255,255,.09);
    --panel:rgba(255,255,255,.035);
    color-scheme:dark;
  }
  *{box-sizing:border-box}
  body{
    margin:0; min-height:100vh; color:var(--txt);
    font-family:'Space Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif;
    background:
      radial-gradient(1100px 620px at 82% -12%, #341063 0%, transparent 60%),
      radial-gradient(900px 520px at -8% 12%, #0c3142 0%, transparent 55%),
      radial-gradient(700px 700px at 50% 120%, #2a0a3f 0%, transparent 60%),
      var(--ink);
    background-attachment:fixed;
  }
  .wrap{width:100%;margin:0;padding:26px 34px 60px}
  header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:8px}
  .brand{font-size:34px;font-weight:700;letter-spacing:-.8px;line-height:1;
    background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
  .tag{color:var(--mut);font-size:13px;font-weight:500;letter-spacing:.2px}
  .live{display:flex;align-items:center;gap:9px;margin-left:auto;
    background:rgba(255,255,255,.04);border:1px solid var(--line);
    padding:7px 14px;border-radius:999px;font-size:13px;color:var(--mut)}
  .dot{width:9px;height:9px;border-radius:50%;background:#5b5570;transition:.2s}
  .dot.on{background:var(--teal);box-shadow:0 0 12px var(--teal);animation:pulse 1.2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  #elapsed{color:var(--txt);font-variant-numeric:tabular-nums}

  /* equalizer hero */
  .hero{margin:18px 0 22px;border-radius:24px;padding:3px;background:var(--grad2);
    box-shadow:0 18px 60px -20px #7b3ff5aa}
  .hero-inner{background:linear-gradient(180deg,#0e0820,#0a0612);border-radius:21px;
    padding:22px 26px;display:flex;align-items:flex-end;gap:5px;height:150px;overflow:hidden}
  .eq{display:flex;align-items:flex-end;gap:5px;width:100%;height:100%}
  .eq i{flex:1;min-width:3px;height:8%;border-radius:5px 5px 2px 2px;
    background:var(--grad);opacity:.4;transform-origin:bottom;transition:opacity .4s}
  .eq.playing i{opacity:1}

  .grid{display:grid;grid-template-columns:380px 1fr;gap:22px;align-items:start}
  @media(max-width:920px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:22px;
    padding:22px;backdrop-filter:blur(8px);margin-bottom:20px}
  .step{display:flex;align-items:center;gap:11px;margin:0 0 16px}
  .num{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;
    font-size:13px;font-weight:700;color:#fff;background:var(--grad2)}
  .step h2{font-size:14px;margin:0;font-weight:600;letter-spacing:.2px}

  .btnrow{display:flex;gap:11px;flex-wrap:wrap}
  .btn{font:inherit;font-weight:600;font-size:14px;cursor:pointer;border-radius:999px;
    padding:12px 22px;border:1px solid var(--line);background:rgba(255,255,255,.05);
    color:var(--txt);transition:.15s}
  .btn:hover:not(:disabled){background:rgba(255,255,255,.11);transform:translateY(-1px)}
  .btn:disabled{opacity:.3;cursor:not-allowed}
  .btn.go{background:var(--grad);border:none;color:#fff;box-shadow:0 8px 26px -8px #ff2d9b}
  .btn.stop{border-color:#ff5a7a55;color:#ff8fa3}
  input[type=text]{flex:1;min-width:150px;font:inherit;font-size:14px;padding:11px 15px;
    border-radius:999px;border:1px solid var(--line);background:rgba(0,0,0,.35);color:var(--txt)}
  input[type=text]::placeholder{color:#6f6790}
  input[type=text]:focus{outline:none;border-color:var(--vio);box-shadow:0 0 0 3px #7b3ff533}

  .meta{display:flex;gap:9px;flex-wrap:wrap;margin-top:16px}
  .chip{font-size:12px;padding:5px 13px;border-radius:999px;background:rgba(255,255,255,.05);
    border:1px solid var(--line);color:var(--mut)}
  .chip b{color:#fff;font-weight:600}
  .hint{color:#7d7499;font-size:12px;line-height:1.6;margin-top:14px}

  .scenes{display:flex;flex-direction:column;gap:10px;max-height:330px;overflow-y:auto;
    margin:-2px -4px -2px -2px;padding:2px 4px 2px 2px}
  .scene{position:relative;padding:13px 15px 13px 19px;border-radius:15px;
    background:rgba(255,255,255,.04);border:1px solid var(--line);overflow:hidden}
  .scene::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--grad2)}
  .scene .n{font-size:10.5px;letter-spacing:1.4px;text-transform:uppercase;color:var(--mut)}
  .scene .ab{margin-top:4px;font-size:13.5px}
  .scene .a{color:#ff7ac6} .scene .b{color:#5fe6d6} .scene .amp{color:#5b5570;margin:0 6px}
  .empty{color:#6f6790;font-size:13px;padding:10px 2px}

  .publish{width:100%;padding:16px;font:inherit;font-size:15px;font-weight:700;cursor:pointer;
    border:none;border-radius:18px;color:#fff;background:var(--grad);letter-spacing:.3px;
    box-shadow:0 12px 34px -10px #ff2d9b;transition:.15s}
  .publish:hover:not(:disabled){filter:brightness(1.08);transform:translateY(-1px)}
  .publish:disabled{opacity:.5;cursor:not-allowed}
  #result{margin-top:14px;font-size:13.5px;line-height:1.5}
  #result a{color:#5fe6d6;font-weight:700;word-break:break-all}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #ffffff44;
    border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:-2px}
  @keyframes spin{to{transform:rotate(360deg)}}
  details{margin-top:10px} summary{cursor:pointer;color:#6f6790;font-size:12px}
  pre{white-space:pre-wrap;font-size:11px;color:var(--mut);max-height:170px;overflow:auto;
    background:rgba(0,0,0,.4);border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:8px}

  .viz-card{padding:0;overflow:hidden}
  .viz-head{display:flex;align-items:center;gap:11px;padding:18px 22px 14px}
  .viz-frame{padding:0 3px 3px}
  .viz-inner{border-radius:0 0 19px 19px;padding:3px 3px 0;background:var(--grad2)}
  iframe{width:100%;height:720px;border:0;border-radius:18px 18px 16px 16px;background:#010409;display:block}
  #vizhint{padding:0 22px 20px;color:#7d7499;font-size:12px}
</style></head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="brand">Score the Story</div>
      <div class="tag">your voice, scored live by AI — then remixable on Audiotool</div>
    </div>
    <div class="live"><span class="dot" id="livedot"></span>
      <span id="status">idle</span><span id="elapsed"></span></div>
  </header>

  <div class="hero"><div class="hero-inner"><div class="eq" id="eq"></div></div></div>

  <div class="grid">
    <div>
      <div class="card">
        <div class="step"><span class="num">1</span><h2>Tell your story</h2></div>
        <div class="btnrow" style="margin-bottom:12px">
          <button class="btn go"   id="startBtn" onclick="start()">▶&nbsp; Start Telling</button>
          <button class="btn stop" id="stopBtn"  onclick="stop()" disabled>■&nbsp; Stop</button>
        </div>
        <div class="btnrow">
          <input type="text" id="inject" placeholder="…or type a scene to inject"
                 onkeydown="if(event.key==='Enter')inject()">
          <button class="btn" onclick="inject()" id="injectBtn" disabled>Inject</button>
        </div>
        <div class="meta">
          <span class="chip">mode <b id="mode">—</b></span>
          <span class="chip">preset <b id="preset">—</b></span>
          <span class="chip">scenes <b id="count">0</b></span>
        </div>
        <div class="hint">Speak naturally — the AI listens and Magenta&nbsp;RT2 scores the
          mood live. Watch it generate in the panel on the right.</div>
      </div>

      <div class="card">
        <div class="step"><span class="num">2</span><h2>Scenes captured</h2></div>
        <div class="scenes" id="scenes"><div class="empty">No scenes yet.</div></div>
      </div>

      <div class="card">
        <div class="step"><span class="num">3</span><h2>Publish to Audiotool</h2></div>
        <button class="publish" id="pubBtn" onclick="publish()">🎵&nbsp; Publish to Audiotool</button>
        <div id="result"></div>
        <div class="hint">Turns your told story into an editable, multiplayer Audiotool
          project. Share the link — anyone can remix it in the browser.</div>
        <details><summary>show log</summary><pre id="log"></pre></details>
      </div>
    </div>

    <div class="card viz-card">
      <div class="viz-head"><span class="num">♪</span>
        <h2>Magenta RealTime&nbsp;2 — live engine</h2></div>
      <div class="viz-frame"><div class="viz-inner">
        <iframe id="viz" src="about:blank" title="live engine"></iframe>
      </div></div>
      <div id="vizhint">The live generative visualization appears here once you press Start.</div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let running=false, vizLoaded=false;

// Build the equalizer hero bars. Heights are driven in real time by the live
// MRT2 audio level (out_level), streamed from the telemetry server.
const eq=$('eq');
const BARS=64;
const bars=[];
const phase=[];
for(let i=0;i<BARS;i++){
  const b=document.createElement('i');
  eq.appendChild(b); bars.push(b);
  phase.push(Math.random()*6.283);
}
let audioLevel=0, audioTarget=0, audioBright=0.5;
let audioSpec=null;                 // latest real FFT bands (length = BARS)
const barCur=new Array(BARS).fill(0.05);   // per-bar smoothed heights

// Subscribe to the telemetry SSE stream for the real audio level (set up once
// telemetry_port is known). out_level ~ 0..0.4 RMS of what's leaving the speakers.
let audioES=null;
function connectAudio(port){
  if(audioES) return;
  try{
    audioES=new EventSource(`http://localhost:${port}/stream`);
    audioES.onmessage=(e)=>{
      let d; try{ d=JSON.parse(e.data); }catch{ return; }
      if(d.kind==='tick'||d.kind===undefined){
        audioTarget=d.out_level??0;
        if(typeof d.out_bright==='number') audioBright=d.out_bright;
        audioSpec=(Array.isArray(d.spec)&&d.spec.length===BARS)?d.spec:null;
      }
    };
    audioES.onerror=()=>{};   // auto-reconnects
  }catch{}
}

// rAF loop. If we have a real FFT spectrum, each bar maps to an actual frequency
// band (low → high, left → right), scaled by loudness. Otherwise we fall back to
// a synthesized shape so the hero still moves before audio arrives.
function animateEq(t){
  audioLevel += (audioTarget-audioLevel)*0.18;     // smooth the level
  const time=t/260, lvl=Math.min(1, audioLevel*2.6);
  for(let i=0;i<BARS;i++){
    let target;
    if(running && audioSpec){
      // real per-frequency energy, kept visible at low volume via a floor
      target = 0.05 + audioSpec[i]*(0.25+0.75*lvl);
    }else if(running){
      const frac=i/(BARS-1);
      const tilt=0.6+0.8*(audioBright*frac+(1-audioBright)*(1-frac));
      const osc=0.5+0.5*Math.sin(time+phase[i]+i*0.35);
      target = 0.06 + lvl*tilt*(0.45+0.55*osc);
    }else{
      target = 0.05+0.03*(0.5+0.5*Math.sin(time+phase[i]+i*0.35));
    }
    // attack fast, release slower — reads like a real spectrum analyzer
    const k = target>barCur[i] ? 0.5 : 0.16;
    barCur[i] += (target-barCur[i])*k;
    bars[i].style.height=(Math.min(1,barCur[i])*100).toFixed(1)+'%';
  }
  requestAnimationFrame(animateEq);
}
requestAnimationFrame(animateEq);

async function post(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
    body:body?JSON.stringify(body):undefined});
  return r.json();
}
async function start(){ await post('/start'); refresh(); }
async function stop(){ await post('/stop'); refresh(); }
async function inject(){
  const el=$('inject'); const t=el.value.trim(); if(!t)return;
  await post('/inject',{text:t}); el.value='';
}
async function publish(){
  const btn=$('pubBtn'); btn.disabled=true;
  $('result').innerHTML='<span class="spin"></span> Publishing… (arranging score + writing project)';
  const res=await post('/publish');
  btn.disabled=false;
  if(res.url){
    $('result').innerHTML=`✓ Your story is live: <a href="${res.url}" target="_blank">${res.url}</a>`;
  }else{
    $('result').innerHTML=`<span style="color:#ff8fa3">✗ ${res.error||'Publish failed'}</span>`;
  }
  if(res.log)$('log').textContent=res.log;
}

function renderScenes(scenes){
  const box=$('scenes');
  if(!scenes.length){box.innerHTML='<div class="empty">No scenes yet.</div>';return;}
  box.innerHTML=scenes.map((s,i)=>
    `<div class="scene"><div class="n">Scene ${i+1} · ${s.key||'—'}</div>
       <div class="ab"><span class="a">${s.a}</span><span class="amp">/</span><span class="b">${s.b}</span></div>
     </div>`).reverse().join('');
}

async function refresh(){
  let s; try{ s=await (await fetch('/state')).json(); }catch{ return; }
  running=s.running;
  const phase=s.phase||'idle';
  const busy=(phase==='starting'||phase==='stopping');
  $('livedot').className='dot'+(running?' on':'');
  const label={starting:'starting (loading engine)…',stopping:'stopping…',
               error:'error — check terminal',idle:'idle'}[phase]
              || (running?(s.status||'listening…'):'idle');
  $('status').textContent=label;
  $('elapsed').textContent=running?` · ${s.elapsed}s`:'';
  $('mode').textContent=s.mode; $('preset').textContent=s.preset;
  $('count').textContent=s.scene_count;
  $('startBtn').disabled=running||busy;
  $('stopBtn').disabled=!running||busy;
  $('injectBtn').disabled=!running;
  eq.classList.toggle('playing', running||phase==='starting');
  renderScenes(s.scenes||[]);
  // The telemetry dashboard is persistent (started at boot) — load it once, and
  // subscribe to its audio stream so the hero bars react to the real output.
  if(s.telemetry_port && !vizLoaded){
    $('viz').src=`http://localhost:${s.telemetry_port}/`;
    $('vizhint').style.display='none'; vizLoaded=true;
    connectAudio(s.telemetry_port);
  }
}
setInterval(refresh,1000); refresh();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
