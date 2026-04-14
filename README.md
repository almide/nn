# nn — Neural Network primitives for Almide

Transformer + signal-processing library written in Almide, built on the `matrix` stdlib.

Status: preview. Used to drive and validate matrix runtime work; Whisper tiny inference runs end-to-end (native + WASM).

## Modules (`src/`)

| File | Purpose |
|---|---|
| `tensor.almd` | Tensor load helpers (GGML f16/f32/f64 decoders) |
| `gguf.almd` / `ggml_whisper.almd` / `whisper_loader.almd` | GGUF/GGML weight loaders for Whisper |
| `mel.almd` / `wav.almd` / `fft.almd` | Audio preprocessing: WAV decode, mel spectrogram, FFT |
| `activations.almd` | gelu / softmax / layer_norm wrappers |
| `attention.almd` | Single- and multi-head attention, masked variants |
| `transformer.almd` | Encoder / decoder blocks (residual, LN, MLP, attention) |
| `generate.almd` | Autoregressive decode loop |
| `whisper.almd` | Full Whisper encoder+decoder pipeline |
| `tokenizer.almd` | BPE tokenizer for Whisper |

## Benchmarks (`examples/`)

Micro + end-to-end matmul benchmarks used to compare Almide against NumPy f32/f64:

- `_bench_matmul.almd`, `_bench_matmul_small.almd`, `_bench_sweep.almd` — matmul throughput 3²…1024²
- `_bench_matmul_f32*.almd`, `_bench_fused_all.almd` — f32 path and fused `α·A·B`
- `_bench_linear_f32.almd` — chained linear layers
- `_bench_mlp_f32.almd` — full MLP: `gelu(X @ W1 + b1) @ W2 + b2`
- `_bench_attn_f32.almd` — full attention block: LN + QKV + softmax + out
- `_bench_encoder_block.almd` — Whisper encoder block end-to-end
- `_bench_ops_breakdown.almd` — per-op dispatch profiling

Demo:

- `examples/browser_demo/` — Whisper running in the browser via WASM + custom WASI shim

## Running

```bash
almide build src/whisper.almd -o whisper          # native
almide build src/whisper.almd -o whisper.wasm --target wasm
almide run examples/_bench_attn_f32.almd
almide test                                       # nn/src tests
```

## Depends on

- [almide/almide](https://github.com/almide/almide) — the compiler and matrix stdlib (f32/f64 paths, cblas dispatch)
