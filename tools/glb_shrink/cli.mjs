// Command-line entry for the glb-shrink pipeline, used by app_zh.py.
// compress.mjs / inspect.mjs / presets.mjs are copied unmodified from
// github.com/lorenhsu1128/glb-shrink (server/), so results match the original tool.
//
//   node cli.mjs inspect  <input.glb>
//   node cli.mjs compress <input.glb> <output.glb> <quality 0-100>
//
// Prints one JSON object to stdout; on failure prints {"error": "..."} and exits 1.
import { readFile, writeFile } from 'node:fs/promises';
import { inspectBuffer } from './inspect.mjs';
import { compressBuffer } from './compress.mjs';
import { resolveSettings, getPresetHint } from './presets.mjs';

async function main() {
  const [cmd, input, output, qualityArg] = process.argv.slice(2);
  if (!input || (cmd !== 'inspect' && cmd !== 'compress')) {
    throw new Error('usage: cli.mjs inspect <in.glb> | compress <in.glb> <out.glb> <quality>');
  }
  const source = await readFile(input);

  if (cmd === 'inspect') {
    const stats = await inspectBuffer(source);
    return { fileSize: source.byteLength, ...stats };
  }

  if (!output) throw new Error('compress needs an output path');
  const { quality, ...profile } = resolveSettings(qualityArg ?? 50);
  const { buffer, stats } = await compressBuffer(source, profile);
  await writeFile(output, buffer);
  return {
    sourceSize: source.byteLength,
    outputSize: buffer.byteLength,
    quality,
    profile,
    hint: getPresetHint(quality),
    stats,
  };
}

main()
  .then((result) => {
    process.stdout.write(JSON.stringify(result));
  })
  .catch((err) => {
    process.stdout.write(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    process.exit(1);
  });
