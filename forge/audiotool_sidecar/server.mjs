/*
 * server.mjs — Audiotool sidecar (runs in Node, NOT Forge's Python).
 *
 * WHY A SEPARATE PROCESS: the Audiotool Nexus SDK (@audiotool/nexus) is
 * JavaScript-only, and Forge's backend is Python. So — exactly like the SA3
 * sidecar — this is a tiny localhost HTTP service that owns one authenticated
 * Nexus client and does the upload+insert. Forge's AudiotoolBridge (Python)
 * calls it. Uses Node's stdlib http only (no extra deps beyond the SDK).
 *
 * Auth: a Personal Access Token in $AUDIOTOOL_PAT (from https://rpc.audiotool.com/dev).
 *
 * Protocol (localhost JSON):
 *   GET  /health  -> {ok:true, authed:bool}
 *   POST /send {wav_path, display_name?, project?, bpm?}
 *        -> {ok:true, project_url, project_name, sample_name}
 *        uploads the WAV as a sample and inserts it onto the project's timeline.
 *        If `project` is omitted, a new project is created and its name returned
 *        (pass it back on later /send calls to keep adding clips to one project).
 *
 * Run (normally auto-spawned by AudiotoolBridge):
 *   AUDIOTOOL_PAT=... PORT=8091 node server.mjs
 */

import http from "node:http";
import { readFile } from "node:fs/promises";
import { createAudiotoolClient, createPATAuth } from "@audiotool/nexus";
import { createDiskWasmLoader } from "@audiotool/nexus/node";

const PORT = Number(process.env.PORT || 8091);
const HOST = process.env.HOST || "127.0.0.1";
const PAT = process.env.AUDIOTOOL_PAT || "";
const STUDIO = "https://beta.audiotool.com/studio?project=";

let client = null;   // the one authenticated Nexus client (created lazily)

async function getClient() {
  if (client) return client;
  if (!PAT) throw new Error("AUDIOTOOL_PAT is not set");
  client = await createAudiotoolClient({
    auth: createPATAuth(PAT),
    wasm: createDiskWasmLoader(),
  });
  return client;
}

/** Accept a project URL (…?project=uuid), a "projects/uuid" name, or a bare uuid
 *  and return the canonical "projects/uuid" name. */
function normalizeProject(input) {
  const s = String(input || "").trim();
  if (!s) throw new Error("project is required");
  const m = s.match(/[?&]project=([^&\s]+)/);
  const id = m ? m[1] : s.replace(/^projects\//, "");
  return id.startsWith("projects/") ? id : `projects/${id}`;
}

/** Read a project's musical context (tempo + time signature) WITHOUT needing any
 *  track — these live on the project-level `config` entity. */
async function projectInfo({ project }) {
  const at = await getClient();
  const projectName = normalizeProject(project);
  const nexus = await at.open(projectName);
  let tempo = 120, num = 4, den = 4;
  let url = STUDIO + projectName.replace("projects/", "");
  try {
    await nexus.start();
    url = nexus.dawUrl || url;
    const cfgs = nexus.queryEntities.ofTypes("config").get();
    if (cfgs.length) {
      const c = cfgs[0];
      tempo = Math.round(c.fields.tempoBpm.value) || 120;
      num = c.fields.signatureNumerator.value || 4;
      den = c.fields.signatureDenominator.value || 4;
    }
  } finally {
    await nexus.stop();
  }
  return { ok: true, project_name: projectName, project_url: url,
           tempo_bpm: tempo, signature_num: num, signature_den: den };
}

/** Minimal RIFF/WAVE PCM duration (seconds) so we can insert without waiting
 *  for server-side transcoding (upload.ready). Works for Forge's 16-bit PCM. */
function wavDurationSeconds(buf) {
  if (buf.length < 44 || buf.toString("ascii", 0, 4) !== "RIFF") return 0;
  const byteRate = buf.readUInt32LE(28) || 1;
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = buf.toString("ascii", off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === "data") return size / byteRate;
    off += 8 + size + (size & 1);
  }
  return 0;
}

async function sendClip({ wav_path, display_name, project, bpm }) {
  if (!wav_path) throw new Error("wav_path is required");
  const at = await getClient();
  const name = (display_name || "Forge clip").slice(0, 80);

  // 1) upload the WAV and WAIT until it's fully processed (transcoded). Inserting
  //    a not-yet-ready sample is what caused "unable to load sample" on rapid
  //    sends — the DAW had nothing downloadable yet. `ready` also gives us the
  //    authoritative duration/bpm.
  const buf = await readFile(wav_path);
  const upload = await at.samples.upload({
    file: new Blob([buf], { type: "audio/wav" }),
    displayName: name,
    kind: "loop",
    visibility: "unlisted",
    tags: ["forge"],
  });
  if (upload instanceof Error) throw upload;
  const meta = await upload.ready;
  if (meta instanceof Error) throw meta;
  const durationSeconds = meta.durationSeconds || wavDurationSeconds(buf) || 4;

  // 2) ensure a project exists (reuse the one passed in, else make a new one)
  let projectName = project ? normalizeProject(project) : null;
  if (!projectName) {
    const res = await at.projects.createProject({
      project: { displayName: "Forge session" },
    });
    if (res instanceof Error) throw res;
    projectName = res.project?.name;
    if (!projectName) throw new Error("project creation returned no name");
  }

  // 3) open, insert onto a new track, and STOP. stop() guarantees the insert is
  //    synced to the backend before we return AND frees the sync session — without
  //    it, only the first send (a fresh empty project) flushed; later clips never
  //    synced, so they "didn't show up".
  const nexus = await at.open(projectName);
  let url = STUDIO + projectName.replace("projects/", "");
  try {
    await nexus.start();
    url = nexus.dawUrl || url;
    await nexus.modify((t) => {
      t.insertSample(
        { name: upload.name, durationSeconds, bpm: meta.bpm || bpm || 120 },
        { displayName: name },   // attachTo omitted → auto-creates an audio track
      );
    });
  } finally {
    await nexus.stop();   // MUST run so the change syncs and the session is released
  }

  return {
    ok: true,
    project_url: url,
    project_name: projectName,
    sample_name: upload.name,
  };
}

function sendJson(res, code, body) {
  const payload = JSON.stringify(body);
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(payload);
}

const POST_ROUTES = { "/send": sendClip, "/project-info": projectInfo };

const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    return sendJson(res, 200, { ok: true, authed: Boolean(PAT) });
  }
  const handler = req.method === "POST" ? POST_ROUTES[req.url] : undefined;
  if (handler) {
    let body = "";
    req.on("data", (c) => { body += c; if (body.length > 1e6) req.destroy(); });
    req.on("end", async () => {
      try {
        sendJson(res, 200, await handler(JSON.parse(body || "{}")));
      } catch (e) {
        console.error(`[audiotool] ${req.url} failed:`, e?.message || e);
        sendJson(res, 500, { ok: false, error: String(e?.message || e) });
      }
    });
    return;
  }
  sendJson(res, 404, { ok: false, error: "not found" });
});

server.listen(PORT, HOST, () => {
  console.log(`[audiotool] serving on http://${HOST}:${PORT} (authed=${Boolean(PAT)})`);
});
