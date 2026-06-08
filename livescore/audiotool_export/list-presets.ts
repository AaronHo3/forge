/**
 * list-presets.ts — see what real presets/soundfonts the Audiotool library has.
 *
 * Use this to verify what a query actually matches before exporting, or to find
 * a specific instrument you like and copy its preset name/ID.
 *
 *   npm run presets gakki cello      # cello soundfonts for the gakki sampler
 *   npm run presets gakki "double bass"
 *   npm run presets gakki bells
 *   npm run presets heisenberg pad   # synth presets
 *
 * Valid device types include: gakki (soundfont sampler), pulverisateur,
 * heisenberg, space (synths), machiniste/beatbox9 (drums).
 */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { createAudiotoolClient } from "@audiotool/nexus";

const envPath = join(import.meta.dirname, ".env");
if (existsSync(envPath)) process.loadEnvFile(envPath);

async function main() {
  const pat = process.env.AT_PAT;
  if (!pat) throw new Error("missing AT_PAT (set it in .env)");

  const [deviceType, ...rest] = process.argv.slice(2);
  if (!deviceType) {
    console.log('Usage: npm run presets <deviceType> [search text]');
    console.log('  e.g. npm run presets gakki cello');
    process.exit(1);
  }
  const query = rest.join(" ");

  const client = await createAudiotoolClient({ authorization: pat });
  // deviceType is validated by the API; cast to the SDK's expected union.
  const matches = await client.api.presets.list(deviceType as Parameters<typeof client.api.presets.list>[0], query);

  console.log(`\n${matches.length} preset(s) for ${deviceType} "${query}":\n`);
  for (const p of matches.slice(0, 30)) {
    const owner = p.meta.ownerName ? `  by ${p.meta.ownerName}` : "";
    console.log(`  • ${p.meta.displayName}${owner}`);
    console.log(`      id: ${p.meta.name}`);
  }
  if (matches.length > 30) console.log(`  … and ${matches.length - 30} more`);
  console.log("\nTip: in the DAW you can also right-click any preset → \"Copy Preset ID\".");
  process.exit(0);
}

main().catch((err) => {
  console.error("✗", err instanceof Error ? err.message : err);
  process.exit(1);
});
