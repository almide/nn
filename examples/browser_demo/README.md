# Browser Demo — Almide WASM Whisper

Run Whisper inference in the browser. Pure Almide → WASM, no JS ML framework,
no native bindings.

## Files

- `index.html` — UI with file pickers for the WASM module, GGML model, and pre-computed mel
- `wasi_shim.js` — Minimal WASI snapshot_preview1 shim (`fd_write`, `clock_time_get`, `random_get`; file-I/O imports return `EBADF` since `transcribe` operates on in-memory bytes)

## Prepare inputs

1. **WASM module**: `whisper_wasm_entry.wasm` from `nn/examples/`
2. **GGML model**: `ggml-tiny.bin` (74 MB) — download via `whisper.cpp`'s helper:
   ```bash
   curl -L -o ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
   ```
3. **Mel input**: pre-compute outside the browser (computing mel inside WASM exhausts the bump heap on 30 s of audio):
   ```bash
   almide build nn/examples/dump_mel.almd -o /tmp/dump_mel
   /tmp/dump_mel ggml-tiny.bin your_audio.wav your_audio.mel.bin
   ```
   The WAV must be 16 kHz mono PCM.

## Serve

Any static server works; the WASM module needs `application/wasm` MIME.

```bash
cd nn/examples/browser_demo
python3 -m http.server 8080
# open http://localhost:8080/
```

## Performance

On an Apple Silicon laptop the inference loop runs at roughly 1.5 s/token in
WASM (~25 s for 5 tokens, 36 s for 30 tokens). Native Almide (Rust target) is
several times faster. WASM SIMD matmul, quantization, and a Burn/ndarray
backend are tracked in `docs/roadmap/active/whisper-almide.md` Phase 6.

## Why pre-compute the mel?

Computing the log-mel spectrogram inside the WASM heap allocates ~3000 frame
lists, which blows past the bump allocator's 4 GB ceiling. The mel computation
is bandwidth-bound, not compute-bound, so doing it on the JS side (or natively
via `dump_mel`) is both faster and avoids the heap issue. The transformer
itself (where Almide actually shines) runs entirely in WASM.
