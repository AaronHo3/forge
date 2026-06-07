/* wave.js - turns every <audio> into a waveform player.
   Decodes the clip (Web Audio), draws accent-colored peak bars on a canvas with a
   play/seek control and a progress fill. A MutationObserver auto-enhances any audio
   added later (generated clips, game reveals) and re-renders when an element's src
   swaps (the trainer reuses one player). Playback still uses the <audio> element, so
   existing JS that calls .play()/sets .src keeps working - we just hide the native UI.
*/
(function () {
  const PLAY = '<svg width="13" height="14" viewBox="0 0 13 14" fill="currentColor"><path d="M1.5 1l10.5 6-10.5 6z"/></svg>';
  const EQ = '<span class="eq"><i></i><i></i><i></i></span>';   // animated bars while playing
  const DIM = "#39435a";
  const cache = new Map();
  let AC = null;
  const ctx = () => (AC = AC || new (window.AudioContext || window.webkitAudioContext)());

  async function peaks(url, buckets) {
    if (cache.has(url)) return cache.get(url);
    const buf = await (await fetch(url)).arrayBuffer();
    const audio = await ctx().decodeAudioData(buf);
    const data = audio.getChannelData(0);
    const block = Math.floor(data.length / buckets) || 1;
    const out = new Float32Array(buckets);
    let mx = 0;
    for (let i = 0; i < buckets; i++) {
      let m = 0;
      const s = i * block;
      for (let j = 0; j < block; j++) { const v = Math.abs(data[s + j] || 0); if (v > m) m = v; }
      out[i] = m; if (m > mx) mx = m;
    }
    if (mx > 0) for (let i = 0; i < buckets; i++) out[i] = out[i] / mx;
    cache.set(url, out);
    return out;
  }

  const fmt = t => (!isFinite(t) ? "0:00" : (t = Math.max(0, t | 0), Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0")));
  const accentOf = el => getComputedStyle(el).getPropertyValue("--accent").trim() || "#5eead4";

  function rrect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  function enhance(audio) {
    if (audio.__waved || !audio.parentNode) return;
    audio.__waved = true;
    audio.removeAttribute("controls");
    audio.style.display = "none";

    const wrap = document.createElement("div"); wrap.className = "wave-player";
    const btn = document.createElement("button"); btn.type = "button"; btn.className = "wave-play"; btn.innerHTML = PLAY;
    const cv = document.createElement("canvas"); cv.className = "wave-cv";
    const time = document.createElement("span"); time.className = "wave-time"; time.textContent = "0:00";
    wrap.append(btn, cv, time);
    audio.parentNode.insertBefore(wrap, audio);

    let bars = null, hoverX = null;
    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth || 280, H = cv.clientHeight || 38;
      cv.width = W * dpr; cv.height = H * dpr;
      const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      const n = bars ? bars.length : Math.floor(W / 3);
      const bw = W / n;
      const prog = (audio.duration ? audio.currentTime / audio.duration : 0) * W;
      const a = accentOf(wrap);
      for (let i = 0; i < n; i++) {
        const v = bars ? bars[i] : 0.12;
        const h = Math.max(2, v * (H - 4)), x = i * bw, y = (H - h) / 2;
        const played = x < prog;
        const preview = !played && hoverX != null && x <= hoverX;   // hover-scrub fill
        g.fillStyle = (played || preview) ? a : DIM;
        g.globalAlpha = preview ? 0.45 : (bars ? 1 : 0.5);
        rrect(g, x + bw * 0.16, y, Math.max(1, bw * 0.68), h, 1); g.fill();
      }
      g.globalAlpha = 1;
      if (hoverX != null) { g.fillStyle = a; g.fillRect(hoverX - 0.5, 0, 1, H); }   // scrub cursor
    }
    function load() {
      bars = null; draw();
      const url = audio.currentSrc || audio.getAttribute("src");
      if (!url) return;
      peaks(url, Math.max(48, Math.floor((cv.clientWidth || 280) / 3))).then(p => { bars = p; draw(); }).catch(() => {});
    }

    btn.onclick = () => {
      if (audio.paused) { document.querySelectorAll("audio").forEach(o => { if (o !== audio) o.pause(); }); audio.play().catch(() => {}); }
      else audio.pause();
    };
    cv.onclick = e => { if (audio.duration) audio.currentTime = (e.clientX - cv.getBoundingClientRect().left) / cv.clientWidth * audio.duration; };
    cv.onmousemove = e => {
      hoverX = e.clientX - cv.getBoundingClientRect().left;
      time.textContent = fmt((hoverX / (cv.clientWidth || 1)) * (audio.duration || 0));
      draw();
    };
    cv.onmouseleave = () => { hoverX = null; time.textContent = fmt(audio.currentTime); draw(); };
    audio.addEventListener("play", () => { btn.innerHTML = EQ; btn.title = "Pause"; wrap.classList.add("playing"); });
    audio.addEventListener("pause", () => { btn.innerHTML = PLAY; btn.title = "Play"; wrap.classList.remove("playing"); });
    audio.addEventListener("ended", () => { btn.innerHTML = PLAY; btn.title = "Play"; wrap.classList.remove("playing"); draw(); });
    audio.addEventListener("timeupdate", () => { if (hoverX == null) time.textContent = fmt(audio.currentTime); draw(); });
    audio.addEventListener("loadedmetadata", () => { time.textContent = fmt(audio.duration); draw(); });
    new MutationObserver(load).observe(audio, { attributes: true, attributeFilter: ["src"] });
    window.addEventListener("resize", draw);
    load();
  }

  function scan(node) {
    if (node.nodeType !== 1) return;
    if (node.tagName === "AUDIO") enhance(node);
    node.querySelectorAll && node.querySelectorAll("audio").forEach(enhance);
  }
  function start() {
    scan(document.body);
    new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(scan)))
      .observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
