/**
 * audiotool_export/index.ts
 * -------------------------
 * The Node/TypeScript half of the Audiotool integration.
 *
 * Reads an `arrangement-*.json` produced by `audiotool_arranger.py` and writes
 * it into a real, editable, shareable Audiotool project using the Nexus SDK
 * (@audiotool/nexus). All musical decisions were already made in Python — this
 * file only maps notes-in-beats onto Audiotool entities. It owns NO music theory.
 *
 * Flow:
 *   1. Authenticate with a Personal Access Token (PAT).
 *   2. Create a fresh project (or open AT_PROJECT if you'd rather write into one).
 *   3. In a single transaction: spin up one synth + mixer channel per track,
 *      then lay each scene down as a note region on that track's timeline.
 *   4. Print the studio URL — open it in a browser and remix/collaborate live.
 *
 * Run:
 *   npm install
 *   AT_PAT=at_pat_xxx ARRANGEMENT=../arrangement-20260606-080622.json npm start
 *   # optional: also set AT_PROJECT="https://beta.audiotool.com/studio?project=…"
 *   # to write into an EXISTING project instead of creating a new one.
 *
 * Get a PAT at: https://developer.audiotool.com/personal-access-tokens
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { createAudiotoolClient } from "@audiotool/nexus";
import { Ticks } from "@audiotool/nexus/utils";

// Auto-load a local .env (sitting next to this file) so you set AT_PAT once and
// then just run `npm start`. Inline env vars still work and take precedence.
const envPath = join(import.meta.dirname, ".env");
if (existsSync(envPath)) {
  process.loadEnvFile(envPath);
}

// ── Arrangement contract (mirrors audiotool_arranger.py output) ────────────────
interface Note {
  positionBeats: number;
  pitch: number;
  durationBeats: number;
  velocity: number;
}
interface Region {
  trackId: string;
  scene: number;
  key: string;
  displayName: string;
  startBeat: number;
  durationBeats: number;
  colorIndex: number;
  notes: Note[];
}
// `gakki` = soundfont sampler (real instruments); the rest are synths. If a
// gakki soundfont can't be matched for a track, we fall back to this synth so
// the part is still audible rather than silent.
type PresetDevice = "gakki" | "pulverisateur" | "heisenberg" | "space";
const FALLBACK_SYNTH: PresetDevice = "heisenberg";

interface Track {
  id: string;
  synth: PresetDevice;
  displayName: string;
  colorIndex: number;
  presetQuery: string;
}
interface Arrangement {
  title: string;
  tempoBpm: number;
  beatsPerBar: number;
  palette: string;
  tracks: Track[];
  regions: Region[];
}

const beats = (b: number) => Math.round(b * Ticks.Beat);

function loadArrangement(path: string): Arrangement {
  const data = JSON.parse(readFileSync(path, "utf8")) as Arrangement;
  if (!data.tracks?.length || !data.regions?.length) {
    throw new Error(`arrangement has no tracks/regions: ${path}`);
  }
  return data;
}

async function main() {
  const pat = process.env.AT_PAT;
  const arrangementPath = process.env.ARRANGEMENT;
  const existingProject = process.env.AT_PROJECT; // optional

  if (!pat) throw new Error("missing AT_PAT (get one at developer.audiotool.com/personal-access-tokens)");
  if (!arrangementPath) throw new Error("missing ARRANGEMENT (path to arrangement-*.json)");

  const arrangement = loadArrangement(arrangementPath);
  console.log(`Arrangement: "${arrangement.title}"`);
  console.log(`  ${arrangement.tracks.length} tracks · ${arrangement.regions.length} regions · ${arrangement.tempoBpm} BPM`);

  const client = await createAudiotoolClient({ authorization: pat });

  // ── Resolve the target project ────────────────────────────────────────────
  let projectRef = existingProject;
  let studioUrl = existingProject ?? "";
  if (!projectRef) {
    console.log("Creating a new Audiotool project…");
    const result = await client.api.projectService.createProject({
      project: { displayName: arrangement.title },
    });
    if (result instanceof Error) throw new Error(`createProject failed: ${result.message}`);
    if (!result.project) throw new Error("createProject returned no project");
    projectRef = result.project.name; // "projects/<uuid>" — contains the UUID
    studioUrl = result.project.name.replace("projects/", "https://beta.audiotool.com/studio?project=");
    console.log(`  → ${studioUrl}`);
  }

  // ── Pick a real preset per track so the synths sound like instruments, not
  //    factory defaults. One synth = one preset for the whole timeline (Audiotool
  //    can't swap instrument per region), so we match each ROLE, biased by the
  //    story's own instrument words (computed in the Python arranger).
  type PresetMatch = Awaited<ReturnType<typeof client.api.presets.list>>[number];
  interface Resolved { deviceType: PresetDevice; preset?: PresetMatch; }
  const resolved = new Map<string, Resolved>();
  for (const track of arrangement.tracks) {
    let deviceType: PresetDevice = track.synth;
    let preset: PresetMatch | undefined;
    if (track.presetQuery) {
      try {
        const matches = await client.api.presets.list(track.synth, track.presetQuery);
        if (matches.length > 0) {
          preset = matches[0];
          console.log(`  ${track.displayName}: ${track.synth} ← "${preset.meta.displayName}" (soundfont match for "${track.presetQuery}")`);
        } else {
          deviceType = FALLBACK_SYNTH;
          console.log(`  ${track.displayName}: no "${track.presetQuery}" soundfont → ${FALLBACK_SYNTH} synth (audible fallback)`);
        }
      } catch {
        deviceType = FALLBACK_SYNTH;
        console.log(`  ${track.displayName}: preset lookup failed → ${FALLBACK_SYNTH} synth`);
      }
    }
    resolved.set(track.id, { deviceType, preset });
  }

  const doc = await client.createSyncedDocument({ mode: "online", project: projectRef! });
  await doc.start();

  // ── One transaction: build synths, mixer wiring, and all the note regions ──
  await doc.modify((t) => {
    // A synth + dedicated mixer channel per track, so every part is audible.
    const players = new Map<string, any>();
    arrangement.tracks.forEach((track, i) => {
      const r = resolved.get(track.id)!;
      const synth = t.create(r.deviceType, {
        displayName: track.displayName,
        positionX: 100,
        positionY: 200 + i * 180,
      });
      if (r.preset) t.applyPresetTo(synth, r.preset);
      const channel = t.create("mixerChannel", {});
      t.create("desktopAudioCable", {
        fromSocket: synth.fields.audioOutput.location,
        toSocket: channel.fields.audioInput.location,
      });
      const noteTrack = t.create("noteTrack", {
        player: synth.location,
        orderAmongTracks: (i + 1) * 100,
      });
      players.set(track.id, noteTrack);
    });

    // Each scene becomes one region per track, placed on the shared timeline.
    for (const region of arrangement.regions) {
      const noteTrack = players.get(region.trackId);
      if (!noteTrack) continue;
      const collection = t.create("noteCollection", {});
      const dur = beats(region.durationBeats);
      t.create("noteRegion", {
        track: noteTrack.location,
        collection: collection.location,
        region: {
          positionTicks: beats(region.startBeat),
          durationTicks: dur,
          loopDurationTicks: dur,
          loopOffsetTicks: 0,
          collectionOffsetTicks: 0,
          colorIndex: region.colorIndex,
          displayName: region.displayName,
          isEnabled: true,
        },
      });
      for (const n of region.notes) {
        t.create("note", {
          collection: collection.location,
          positionTicks: beats(n.positionBeats),
          durationTicks: beats(n.durationBeats),
          pitch: n.pitch,
          velocity: n.velocity,
        });
      }
    }
  });

  const noteCount = arrangement.regions.reduce((s, r) => s + r.notes.length, 0);
  console.log(`✓ Wrote ${arrangement.regions.length} regions · ${noteCount} notes into Audiotool.`);
  console.log(`\n🎵 Open & remix (multiplayer): ${studioUrl}\n`);

  // The synced document has no stop() yet (SDK is pre-1.0); give the last
  // transaction a moment to flush to the backend, then exit cleanly.
  await new Promise((r) => setTimeout(r, 2500));
  process.exit(0);
}

main().catch((err) => {
  console.error("✗ Export failed:", err instanceof Error ? err.message : err);
  process.exit(1);
});
