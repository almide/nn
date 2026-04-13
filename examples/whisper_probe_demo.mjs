#!/usr/bin/env node
import { WASI } from 'node:wasi';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WASM_PATH = resolve(__dirname, 'whisper_wasm_probe.wasm');

async function main() {
  const [modelPath] = process.argv.slice(2);
  const wasmBuf = await readFile(WASM_PATH);
  const modelBuf = await readFile(modelPath);

  const wasi = new WASI({ version: 'preview1', args: [], env: {}, preopens: {} });
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.getImportObject());
  wasi.initialize(instance);

  const exports = instance.exports;
  const memory = exports.memory;
  const alloc = exports.__alloc;
  const probe = exports.probe;
  const probe_pure = exports.probe_pure;
  const probe2 = exports.probe2;
  const probe3 = exports.probe3;
  const probe_const = exports.probe_const;

  console.log(`[probe] modelBuf.length=${modelBuf.length}`);
  console.log(`[probe] memory pages: ${memory.buffer.byteLength / 65536}`);
  const modelPtr = alloc(4 + modelBuf.length);
  console.log(`[probe] modelPtr=${modelPtr}, memory pages after: ${memory.buffer.byteLength / 65536}`);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.length, true);
  new Uint8Array(memory.buffer).set(modelBuf, modelPtr + 4);
  // Read back for verification
  console.log(`[probe] readback len: ${new DataView(memory.buffer).getInt32(modelPtr, true)}`);
  console.log(`[probe] readback first 4 bytes at ptr+4: 0x${new DataView(memory.buffer).getUint32(modelPtr + 4, true).toString(16)}`);

  // Effect fn returns are Result[T, String] = i32 ptr to [tag:i32][payload].
  // For Int payload, payload is i64 at offset 8 (alignment) or offset 4.
  function unwrapResultInt(ptr) {
    const v = new DataView(memory.buffer);
    const tag = v.getInt32(ptr, true);
    if (tag !== 0) throw new Error('Err');
    // Try i64 at offset 4 first, then 8
    return v.getBigInt64(ptr + 4, true);
  }
  function unwrapResultStr(ptr) {
    const v = new DataView(memory.buffer);
    const tag = v.getInt32(ptr, true);
    if (tag !== 0) throw new Error('Err: ' + readString(v.getInt32(ptr + 4, true)));
    return readString(v.getInt32(ptr + 4, true));
  }
  function readString(ptr) {
    const v = new DataView(memory.buffer);
    const len = v.getInt32(ptr, true);
    return new TextDecoder().decode(new Uint8Array(memory.buffer, ptr + 4, len));
  }
  console.log(`[probe_const: effect fn → Result] ptr=${probe_const()}`);
  console.log(`[probe_pure: bytes.len pure] = ${probe_pure(modelPtr)}`);
  console.log(`[probe1 effect ptr] = ${probe(modelPtr)} → unwrap: ${unwrapResultInt(probe(modelPtr))}`);
  console.log(`[probe2 effect ptr] = ${probe2(modelPtr)} → unwrap: 0x${unwrapResultInt(probe2(modelPtr)).toString(16)} (expected 0x67676d6c)`);
  const p3 = probe3(modelPtr);
  console.log(`[probe3 ptr] = ${p3} → unwrap: ${unwrapResultInt(p3)} (expected 51864)`);
  const pvc = exports.probe_vocab_count(modelPtr);
  console.log(`[probe_vocab_count] → unwrap: ${unwrapResultInt(pvc)} (expected 50257)`);
  const pft = exports.probe_first_token(modelPtr);
  console.log(`[probe_first_token] → unwrap: "${unwrapResultStr(pft)}"`);

  console.log(`[probe_load_filter] starting...`);
  const pf = exports.probe_load_filter(modelPtr);
  console.log(`[probe_load_filter] → unwrap: ${unwrapResultInt(pf)} (expected 80)`);

  console.log(`[probe_load_conv1] starting...`);
  const pc = exports.probe_load_conv1(modelPtr);
  console.log(`[probe_load_conv1] → unwrap: ${unwrapResultInt(pc)} (expected 384)`);

  console.log(`[probe_load_weights] starting...`);
  const pw = exports.probe_load_weights(modelPtr);
  console.log(`[probe_load_weights] → unwrap: ${unwrapResultInt(pw)} (expected 51864)`);

  // Mel test needs audio
  const audioPath = process.argv[3];
  if (audioPath) {
    const audioBuf = await readFile(audioPath);
    const audioPtr = alloc(4 + audioBuf.length);
    new DataView(memory.buffer).setInt32(audioPtr, audioBuf.length, true);
    new Uint8Array(memory.buffer).set(audioBuf, audioPtr + 4);
    console.log(`[probe_mel] starting...`);
    const t0 = Date.now();
    const pm = exports.probe_mel(modelPtr, audioPtr);
    console.log(`[probe_mel] → unwrap: ${unwrapResultInt(pm)} (expected ~3000) in ${Date.now() - t0} ms`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
