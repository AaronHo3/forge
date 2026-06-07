/*
 * jam-client.js - shared real-time audio for MRT2 continuous generation.
 *
 * JamPlayer: gapless Web Audio playback of int16 interleaved-stereo PCM (48k).
 *   Reusable on any WebSocket (Live Morph, Journey, Live Trainer use /ws/jam;
 *   the Classroom uses /ws/classroom). Just feed() it the binary chunks.
 *
 * JamClient: JamPlayer + a /ws/jam connection + control/prompt/save/stop helpers.
 */
function JamPlayer() {
  let ctx = null, nextTime = 0;
  const LEAD = 0.12;   // scheduling lead for gapless playback (small for low latency)
  return {
    async start() {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      await ctx.resume();
      nextTime = 0;
    },
    feed(arrayBuf) {
      if (!ctx) return;
      const i16 = new Int16Array(arrayBuf);
      const frames = i16.length / 2;
      const buf = ctx.createBuffer(2, frames, 48000);
      const L = buf.getChannelData(0), R = buf.getChannelData(1);
      for (let i = 0; i < frames; i++) { L[i] = i16[2 * i] / 32768; R[i] = i16[2 * i + 1] / 32768; }
      const now = ctx.currentTime;
      if (nextTime < now + 0.02) nextTime = now + LEAD;   // (re)prime if we fell behind
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination); src.start(nextTime);
      nextTime += buf.duration;
    },
    stop() { try { if (ctx) ctx.close(); } catch (e) {} ctx = null; },
  };
}

function JamClient() {
  let ws = null, playing = false, jsonCb = null, started = false;
  const player = JamPlayer();

  function wsUrl() {
    return (location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/ws/jam";
  }
  function send(obj) { if (playing && ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

  return {
    get playing() { return playing; },
    async start(initMsg, onJson) {
      await player.start();
      started = false; jsonCb = onJson;
      return new Promise((resolve, reject) => {
        ws = new WebSocket(wsUrl());
        ws.binaryType = "arraybuffer";
        ws.onopen = () => { playing = true; ws.send(JSON.stringify(initMsg)); resolve(); };
        ws.onmessage = (e) => {
          if (typeof e.data === "string") { if (jsonCb) jsonCb(JSON.parse(e.data)); }
          else {
            if (!started) { started = true; if (jsonCb) jsonCb({ type: "playing" }); }
            player.feed(e.data);
          }
        };
        ws.onclose = () => { playing = false; if (jsonCb) jsonCb({ type: "closed" }); };
        ws.onerror = (err) => { playing = false; reject(err); };
      });
    },
    control(p) { send({ action: "control", ...p }); },
    prompt(p) { send({ action: "prompt", ...p }); },
    save(seconds) { send({ action: "save", seconds }); },
    stop() {
      try { send({ action: "stop" }); } catch (e) {}
      try { if (ws) ws.close(); } catch (e) {}
      player.stop();
      playing = false;
    },
  };
}
