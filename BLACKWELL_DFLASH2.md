# Blackwell DFlash2 fork

This repository is the tested runtime for
[`sirus/Qwen3.8-27B-heretic-ara-MXFP6-MXFP8-DFlash2-GGUF`](https://huggingface.co/sirus/Qwen3.8-27B-heretic-ara-MXFP6-MXFP8-DFlash2-GGUF).

It packages two ABI-compatible llama.cpp trees:

- The repository root provides the DFlash2-capable `llama-server` frontend.
- [`runtime/`](runtime/) provides the Blackwell CUDA runtime and advanced GGUF tooling used by the server through `LD_PRELOAD`.

This layout preserves the exact combination used for the published model. It is an experimental research fork, not an upstream llama.cpp release.

The server tree is based on DFlash2 PR snapshot `5ecbe1ac1`. The runtime tree is based on advanced-gguf-quantizer snapshot `8b87f1b64` plus the published DFlash2 handoff, MXFP6/MXFP8, and TurboQuant integration changes.

The inherited upstream workflows are retained under `.github/upstream-workflows-disabled/` for reference. They are not enabled for this experimental bundle.

## Included work

- DFlash2 support based on [llama.cpp PR 27342](https://github.com/ggml-org/llama.cpp/pull/27342).
- GPU-resident DFlash2 target feature handoff with fused projection and KV injection.
- Native Blackwell NVFP4, experimental MXFP6_E2M3, and OCP MXFP8 weights.
- MXFP8 and experimental TurboQuant KV cache types.
- CUDA flash attention for the custom quantized formats.
- The advanced GGUF quantizer and its recipe, inspection, and benchmark tools.

## Build

CUDA 13.x and a Blackwell GPU are the primary tested configuration.

```bash
./build-blackwell-dflash2.sh
```

The script builds:

- `build/bin/llama-server`
- `build/bin/libllama-custom-runtime-abi.so`
- `runtime/build-blackwell/bin/libllama.so.0`
- the custom GGML CUDA libraries

## Download the model

```bash
hf download \
  sirus/Qwen3.8-27B-heretic-ara-MXFP6-MXFP8-DFlash2-GGUF \
  Qwen3.8-27B-heretic-ara-MXFP6-MXFP8.gguf \
  Qwen3.8-27B-DFlash2-MXFP6_E2M3.gguf \
  --local-dir models
```

## Run

```bash
./run-qwen3.8-heretic-dflash2.sh
```

Defaults:

- one slot;
- 262144-token context;
- 4096 batch and ubatch;
- MXFP8 K/V cache;
- three DFlash2 draft tokens;
- OpenAI-compatible server on port 9080.

All important paths and runtime settings can be overridden with environment variables. For example:

```bash
LLAMA_PORT=19080 CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 \
  ./run-qwen3.8-heretic-dflash2.sh
```

`turbo_mxfp8`, `turbo2`, `turbo3`, and `turbo4` are available for KV cache experiments. Plain MXFP8 remains the default because it produced better DFlash2 end-to-end throughput in the tested workload.

## Status and limitations

- MXFP6_E2M3 and the custom GGUF type IDs are experimental and require this runtime.
- The published target is text-only even though the upstream architecture can include vision components.
- Quality evaluation is incomplete. The artifact passed structural inspection, MTP preservation checks, completion smoke tests, and long-context runtime tests, but no final PPL/KLD result is claimed.
- The DFlash2 draft is derived from [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

## Attribution

This repository retains the upstream llama.cpp license and notices. Important sources include:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [incoai DFlash2](https://inco.ai/blog/dflash2/)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- [michaelw9999/advanced-gguf-quantizer](https://github.com/michaelw9999/advanced-gguf-quantizer)
- [MIT HAN Lab Four Over Six](https://github.com/mit-han-lab/fouroversix)

See [`runtime/NOTICE.md`](runtime/NOTICE.md) for the full advanced quantization notice.
