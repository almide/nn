#!/usr/bin/env node
// Node.js harness for the Almide WASM Whisper module.
//
// Usage:  node whisper_node_demo.mjs <model.bin> <mel.bin>
//
// `mel.bin` is a flat (frames × 80) f64 LE buffer produced by
// dump_mel.almd. Computing mel inside WASM exhausts the bump heap
// because per-frame List allocations accumulate; pre-computing it
// outside lets WASM do only the heavy transformer math.

import { WASI } from 'node:wasi';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WASM_PATH = resolve(__dirname, 'whisper_wasm_entry.wasm');

async function main() {
  const [modelPath, melPath] = process.argv.slice(2);
  if (!modelPath || !melPath) {
    console.error('Usage: node whisper_node_demo.mjs <model.bin> <mel.bin>');
    process.exit(1);
  }

  console.log(`[node] loading WASM ${WASM_PATH}`);
  const wasmBuf = await readFile(WASM_PATH);
  const modelBuf = await readFile(modelPath);
  const melBuf = await readFile(melPath);
  console.log(`[node] model ${(modelBuf.length / 1024 / 1024).toFixed(1)} MB`);
  console.log(`[node] mel   ${melBuf.length} bytes (= ${melBuf.length / 8 / 80} frames × 80 mels)`);

  const wasi = new WASI({ version: 'preview1', args: [], env: {}, preopens: {} });
  const t0 = Date.now();
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.getImportObject());
  wasi.initialize(instance);
  console.log(`[node] WASM instantiated in ${Date.now() - t0} ms`);

  const exports = instance.exports;
  const memory = exports.memory;
  const alloc = exports.__alloc;
  const transcribe = exports.transcribe;

  function writeBytes(buf) {
    // alloc returns i32; cast to u32 in case heap pointer crosses 2GB.
    const ptr = alloc(4 + buf.length) >>> 0;
    new DataView(memory.buffer).setInt32(ptr, buf.length, true);
    new Uint8Array(memory.buffer).set(buf, ptr + 4);
    return ptr;
  }

  function readString(ptr) {
    const v = new DataView(memory.buffer);
    const len = v.getInt32(ptr, true);
    return new TextDecoder('utf-8').decode(new Uint8Array(memory.buffer, ptr + 4, len));
  }

  // Effect fn returns Result[String, String] = [tag:i32][payload_ptr:i32].
  function unwrapResultString(rawPtr) {
    const ptr = rawPtr >>> 0;
    const v = new DataView(memory.buffer);
    const tag = v.getInt32(ptr, true);
    const payload = v.getInt32(ptr + 4, true) >>> 0;
    if (tag !== 0) {
      throw new Error(`Almide returned Err: ${readString(payload)}`);
    }
    return readString(payload);
  }

  const t1 = Date.now();
  const modelPtr = writeBytes(modelBuf);
  const melPtr = writeBytes(melBuf);
  console.log(`[node] memory ready (model+mel) in ${Date.now() - t1} ms`);

  const melFrames = melBuf.length / 8 / 80;
  console.log(`[node] calling transcribe(model, mel, frames=${melFrames}, max=5)`);
  const t2 = Date.now();
  const maxNew = BigInt(process.argv[4] ?? 20);
  const resultPtr = transcribe(modelPtr, melPtr, BigInt(melFrames), maxNew);
  const elapsed = Date.now() - t2;
  const text = unwrapResultString(resultPtr);

  console.log(`[node] transcribe returned in ${elapsed} ms (${(elapsed / 1000).toFixed(1)} s)`);
  console.log('');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`Transcription: "${text}"`);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
