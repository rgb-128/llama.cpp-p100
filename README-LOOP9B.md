# SenseNova-Loop-9B on an Nvidia Tesla P100 (sm_60): 34-36 tok/s on one card

This branch runs SenseTime's **SenseNova-Loop-9B** (text + vision) on a single 16 GB Tesla P100.
Upstream llama.cpp did not support the model yet (looped decoder + two-stream mHC). This branch ports
SenseTime's reference implementation onto a P100-tuned llama.cpp fork and fuses the mHC steps into
three GPU ops.

| build (one P100, Q4_K_M, same prompt) | generation |
|---|---|
| SenseTime's reference llama.cpp branch | 6.9 tok/s |
| this branch (fused mHC ops) | 34.0 tok/s (36.5 peak on a short prompt) |

## What changed
1. Port of the SenseNova-Loop architecture, mHC and MTP head (commit d4e9e4478), including the fused
   `qkv_gate` / `beta_alpha` weight layout the official GGUF uses.
2. Three fused ops, CPU + CUDA: `ggml_mhc_sinkhorn2`, `ggml_mhc_pre`, `ggml_mhc_merge` (commit 62f5f2b41).
   Unfused, one token needed ~2,150 tiny kernel launches and the card sat at 70-100 W doing 8 tok/s.
   Tests: `tests/test-mhc-fused.cpp`, `tests/test-mhc-sinkhorn2.cpp`, `test-backend-ops -o MHC_SINKHORN2`.
3. Not used: MTP speculative drafts (slower on this card, every setting tried) and CUDA graphs (no gain).

## Build (CUDA 12.x)
```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release -DLLAMA_UI_HF_BUCKET=none/none
cmake --build build -j --target llama-server
```

## Run
Model files from `sensenova/SenseNova-Loop-9B-GGUF`: `SenseNova-Loop-9B-Q4_K_M.gguf` (+ `mmproj-SenseNova-Loop-9B-F16.gguf` for images).
```
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build/bin/llama-server \
  -m SenseNova-Loop-9B-Q4_K_M.gguf --mmproj mmproj-SenseNova-Loop-9B-F16.gguf --image-min-tokens 1024 \
  -ngl 99 -fa on -c 98304 -ctk q8_0 -ctv q8_0 -np 1 --jinja --port 8080
```
VRAM: ~6.4 GB at 16K context, ~9.4 GB with vision and 98K context.

## Credits
SenseTime / SenseNova (model, reference implementation: blueswhen/llama.cpp `sensenova-loop`), ggml-org/llama.cpp,
Kmic-68/llama.cpp `p100-optimizations` (Pascal kernels this fork builds on), shinbunbun (P100 patch 01).
See NOTICE.md. MIT license, as upstream.
