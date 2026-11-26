# llama.cpp-p100 (p100-universal-2026)

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) synced to Nov 26, 2026 (`fa9f928f2`) with Pascal `sm_60` and `sm_61` CUDA kernels, multi-GPU collectives, and installer scripts. Builds on work from [Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp) and [junkgpulab/llama.cpp-p100](https://github.com/junkgpulab/llama.cpp-p100). Upstream README is below the line.

## Install

Requires Linux x86_64, NVIDIA driver 580.x, and CUDA Toolkit 12.x. CUDA 13 dropped native `sm_60` and `sm_61` support. Works on single-CPU desktops or multi-socket hosts with 1 to 4 Pascal GPUs.

```bash
git clone -b p100-universal-2026 https://github.com/rgb-128/llama.cpp-p100.git
cd llama.cpp-p100
./install-pascal.sh --install-deps
```

`install-pascal.sh` detects your GPU compute capability (`60` for P100, `61` for P40/P4/1080 Ti/Titan Xp), finds CUDA 12.x, and builds single-arch binaries into `build-p100/bin/` or `build-sm61/bin/`.

Run a model:

```bash
./scripts/run-pascal.sh -m /path/to/model.gguf
./scripts/run-pascal.sh -m /path/to/model.gguf -d /path/to/draft.gguf
./scripts/run-pascal.sh -m /path/to/model.gguf --gpus 2
```

`run-pascal.sh` picks single-GPU mode for 1 card, `-sm tensor` for 2 or 4 cards, and `-sm layer` for 3 cards. On multi-socket hosts it binds CPU and memory to the GPUs' NUMA node when all selected cards share one socket. If a model architecture does not implement tensor split, it retries with `-sm layer` automatically.

## Supported hardware

| Cards | Arch | Multi-GPU mode | What runs |
| :--- | :--- | :--- | :--- |
| 1-4x Tesla P100 | `sm_60` | 1: single, 2/4: tensor, 3: layer | FP16x2 MoE grouped GEMM, FP16x2 and XMAD mat-vec, NVFP4, IQ1-IQ3, fused GDN gate, Pascal AllReduce |
| 1-4x Tesla P40, P4, GTX 1080 Ti, Titan Xp | `sm_61` | 1: single, 2/4: tensor, 3: layer | Native `__dp4a`, SMEM bank-split IQ1-IQ3, branchless `ue4m3`, Pascal AllReduce |

Any x86_64 CPU works. P2P BAR reads are self-tested at startup and fall back to mapped host RAM if the board does not allow P2P.

## Benchmarks

Measured with `llama-bench -fa 1` using default settings and live `llama-server` chat at `temp=0` on Tesla P100-PCIE-16GB cards over PCIe Gen3 x16. All models use Q4 or larger quants. All speeds are tokens per second (`tps`).

| Model | Size | Setup | Prefill 512 (tps) | Prefill 2048 (tps) | Plain Decode (tps) | Live Decode (tps) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| LFM2.5-24B-A2B UD-Q4_K_XL | 13.9 GiB | 1x P100 | 1365.0 | 1365.7 | 120.6 | 115.3-115.6 |
| LFM2.5-24B-A2B UD-Q4_K_XL | 13.9 GiB | 4x P100 tensor | 2082.7 | 2030.0 | 131.0 | - |
| Nemotron-3.5-Bolt-E4B UD-Q8_K_XL | 7.6 GiB | 1x P100 + DFlash | 1975.8 | 2000.6 | 60.8 | 101.4-116.3 |
| Nemotron-3.5-Bolt-E4B UD-Q8_K_XL | 7.6 GiB | 2x P100 tensor + DFlash | 2576.5 | 2642.9 | 75.9 | 108.6-125.0 |
| Gemma-4-26B-A4B-it UD-Q4_K_XL | 15.8 GiB | 2x P100 tensor + MTP | 1345.7 | 1286.1 | 79.2 | 108.1-114.3 |
| Gemma-4-26B-A4B-it UD-Q4_K_XL | 15.8 GiB | 4x P100 tensor | 1512.4 | 1423.2 | 81.0 | - |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 20.8 GiB | 2x P100 tensor | 1480.5 | 1440.0 | 86.9 | 86.6-88.4 |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 20.8 GiB | 2x P100 layer | 1107.8 | 1499.4 | 77.0 | - |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 20.8 GiB | 4x P100 tensor | 1404.4 | 1339.6 | 80.4 | - |
| Flint-39B-A2.5B UD-Q4_K_XL | 23.3 GiB | 2x P100 layer | 929.5 | 1203.7 | 77.9 | 77.5 |
| Qwen3.8-27B UD-Q4_K_XL | 16.4 GiB | 2x P100 tensor + MTP | 400.6 | 395.1 | 36.5 | 43.7-54.7 |
| Qwen3.8-27B UD-Q4_K_XL | 16.4 GiB | 4x P100 tensor + MTP | 533.4 | 521.7 | 43.8 | 42.9-56.1 |
| Gemma-4-31B-it UD-Q4_K_XL | 17.5 GiB | 2x P100 tensor + MTP | 329.1 | 314.0 | 32.0 | 42.2-57.8 |
| Gemma-4-31B-it UD-Q4_K_XL | 17.5 GiB | 4x P100 tensor + MTP | 461.4 | 441.6 | 40.4 | 60.4-68.8 |
| Inkling-2-Mini UD-Q4_K_XL | 16.4 GiB | 4x P100 layer + DSpark | 419.3 | 414.5 | 21.2 | 23.9-26.9 |

Plain Decode is `llama-bench -n 128` without a drafter. Live Decode uses `llama-server` prose and code prompts with MTP width 4 or DFlash/DSpark width 3 where listed in Setup.

### Kernel microbenchmarks

Measured on 1x Tesla P100 with `test-backend-ops perf` (`m=4096, k=14336` unless noted). All pass CPU reference checks.

| Kernel | Before | After | Gain |
| :--- | ---: | ---: | ---: |
| NVFP4 `n=1` | 100.9 us | 83.5 us | +20.8% |
| NVFP4 `n=4` | 356.6 us | 191.6 us | +86.1% |
| NVFP4 `n=5` | 418.5 us | 210.0 us | +99.3% |
| MXFP4 MoE `m=2880, k=2880`, `n=1` | 66.1 us | 45.7 us | +44.7% |
| MXFP4 MoE `m=2880, k=2880`, `n=4` | 296.7 us | 196.1 us | +51.3% |
| IQ3_S `m=8704, k=5120`, `n=1` | 92.4 us | 75.6 us | +22.3% |
| IQ3_S `n=5` | 293.8 us | 194.8 us | +50.8% |
| IQ2_XS `n=5` | 290.8 us | 213.7 us | +36.1% |
| IQ2_XS MMQ prefill `n=512` | 5.37 ms | 4.08 ms | +31.4% |

## What changed

- **Upstream sync:** Merged `ggml-org/llama.cpp` `fa9f928f2` so Qwen3.8, Qwen3.6, Gemma-4, LFM2.5, Nemotron-3.5-Bolt, Inkling-2, and Flint all run with Pascal kernels intact.
- **Native FP16x2 MoE prefill:** `mmid-f16-p100.cu` and `mmq.cu` replace emulated-DP4A `MMQ` on `sm_60` for `MUL_MAT_ID` with a zero-sync FP16-product / FP32-accumulate grouped GEMM at `STACK:0`. It unpacks all 15 primary quant types on the fly and pre-dequantizes IQ1, IQ2, FP8, and future ggml quant types automatically, lifting MoE prefill by +12% to +70% across LFM2.5, Gemma-4-26B, Qwen3.6, Flint, and Inkling-2.
- **Default single-token Pascal CUDA graphs:** `ggml-cuda.cu` enables `GGML_CUDA_GRAPHS_PRE_VOLTA=3` by default on `sm_60` and `sm_61` and fixes the `MUL_MAT_ID` and F16/BF16/F32 batch-size checks so MoE and dense models both capture single-token decode graphs. 4x LFM2.5 decode climbs from 38.4 to 131.0 tps and 4x Gemma-4-31B climbs from 23.2 to 40.4 tps.
- **Tied LM head sharding:** `src/llama-model.cpp` shards duplicated tied `output` heads across GPUs in `-sm tensor` instead of mirroring them on every card. This cuts Gemma-4 262K vocab head latency by 2x to 4x and frees 450 to 690 MB VRAM per GPU.
- **Coalesced narrow `GET_ROWS`:** `getrows.cu` packs multiple rows per 256-thread block when row width is up to 32 elements or 16-byte vectors. Inkling-2 relative-position bias lookup drops from 505 ms to ~8 ms, lifting `pp2048` by +70.6%.
- **Dense prefill fixes:** `gemm-fold.cu` and `ggml-cuda.cu` disable dead 4-way `xchg_chunks` slicing when internal AllReduce is active, adding 5 to 7% on 2x and 4x dense prefill, and extend fused GLU prescale to GeGLU and arbitrary FFN widths past K=11264.
- **2x/4x Pascal AllReduce:** `allreduce.cu` adds zero-sync mapped-RAM DDIO and BAR1-read P2P kernels at `STACK:0`, auto-enabled on 2 and 4 Pascal cards. Beats NCCL on 4x P100 Qwen3.8-27B by +28% `pp512`, +41% `pp2048`, and +5% `tg128`, and by +30% on 2x P100. Override with `GGML_CUDA_ALLREDUCE=nccl|internal|none`.
- **NVFP4 and branchless `ue4m3`:** `common.cuh`, `vecdotq-p100.cuh`, and `mmvq.cu` replace CUDA's software FP8 conversion on pre-Ada GPUs with 4 integer/float ops and add `vec_dot_nvfp4_q8_1_p100`.
- **IQ1-IQ3, MXFP4, Q2_0:** `vecdotq-p100.cuh` and `mmvq.cu` use bank-split SMEM codebooks and branchless sign handling on `sm_60` and `sm_61`.
- **Fused Q8_0 GDN gate and alignment fix:** `mmvq-f16.cu` cuts 240 to 288 launches per token on Qwen3.8-27B, and `gdn-chunked.cu` passes all 103 `GATED_DELTA_NET` backend tests.

Notes:
- Build single-arch only (`60` or `61`). Multi-arch builds fall back to upstream kernels.
- On 4-GPU chassis where ~500 W sustained load triggers PSU `BD PROCHOT` CPU throttling, `sudo nvidia-smi -ac 715,1164` keeps full 715 MHz HBM2 at ~85-95 W/card without throttling the CPU.
- More technical details: [docs/P100.md](docs/P100.md).

---

## Upstream Fork Baseline (`junkgpulab/llama.cpp-p100`)

**Original baseline:** upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) (master `6050b467c`, b11887, 2026-10-25)
plus the Pascal / sm_60 kernels from **[Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp)** (`p100-optimizations`)
plus junkgpulab's fixes:

- **Padded speculative-verify fix:** unused verify columns could hold garbage, `0 x NaN` then poisoned the recurrent
  state and the model answered "4444..." from the second request on. Pad columns are now zeroed and scale-by-0 writes 0.
- **GDN in-place gather fix:** the gated-delta-net kernel could read a row index whose buffer had already been reused
  (Xid 13 on Qwen3.8-Flash-Next). The index now lives in a private device buffer.
- **Compact KQ mask limited to the qwen35 arch**, so other hybrid graphs (Qwen3.8-Flash-Next) load.
- Opt-in MoE / Flash-Next work: BF16 and K-quant P100 mat-vec kernels, pinned host experts
  (`GGML_CUDA_REGISTER_HOST=1`), per-card MoE cache budget (`LLAMA_MOE_CACHE_MIB_DEV`). All off by default.
- A one-line build preset for sm_60.

The speed comes from Kmic-68 and co's kernels. Patch series and notes: [docs/P100.md](docs/P100.md).

## Speed: it ties Kmic's fork, it does not beat it

Measured by us on one box, 2026-10-27: 2x Tesla P100 16 GB (PCIe), tensor split (`-sm tensor`),
P2P on, Qwen3.8-27B Q6_K, MTP speculative decoding with 4 drafts, p-min 0.2, plus one tuned draft setting, `LLAMA_SPEC_SAMPLE_TEMP=0` (greedy drafts; our first video's recipe had the draft guesser sampling at 1.0), explained in [our second video](https://youtube.com/watch?v=oYLZFgduLDU).
"Kmic" = Kmic-68's `p100-optimizations` tip plus our padded-verify fix, same box, same flags, same session.

| test | this fork | Kmic's fork | reading |
|---|---|---|---|
| 22-prompt head-to-head, overall t/s | 45.2 | 45.4 / 45.9 (two arms) | tie: median per-prompt ratio 0.999, inside Kmic's own run-to-run drift (0.990) |
| 11-question bench, prose / code / 30K-context t/s | 58.9 / 70.5 / 56.5 | 59.8 / 71.4 / 57.3 | ~1.5% behind |
| DFlash draft entry at 30K context, t/s | 44.5 | 41.4 | +7%, from draft acceptance (upstream numerics), not a kernel win |

Single box, few runs; treat differences under ~3% as noise. No numbers at 62K+ context are claimed for this build.

## Build

sm_60 **only**: a multi-arch build silently falls back to upstream's kernels.

**You need** (Linux x86_64; tested on Ubuntu 24.04):

| what | version | why |
|---|---|---|
| NVIDIA driver | 580 series | the last driver branch that supports Pascal; 590+ drops the P100 |
| CUDA toolkit | 12.x (tested 12.9) | **not 13.x**: CUDA 13 cannot compile for sm_60 |
| CMake | 3.23 or newer (tested 3.28) | the build uses `CMakePresets.json` |
| Ninja | any recent (tested 1.12) | the preset's generator |
| gcc / g++ | up to 14 (tested 13.3) | host compiler for nvcc 12.x |
| git | any | |

On Ubuntu: `sudo apt install build-essential cmake ninja-build git`, then CUDA 12.x from NVIDIA
(the `cuda-toolkit-12-9` package, or the runfile with the driver box unticked if you already have driver 580).
Make sure `nvcc` is on your `PATH` and is 12.x:

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH   # adjust to where CUDA 12.x is installed
nvcc --version                               # must say release 12.x
nvidia-smi                                   # must show your P100(s) and driver 580.x
```

Then:

```bash
git clone https://github.com/junkgpulab/llama.cpp-p100
cd llama.cpp-p100
cmake --preset p100                # CUDA on, CMAKE_CUDA_ARCHITECTURES=60, Release, Ninja
cmake --build --preset p100        # llama-server + llama-bench in build-p100/bin/
./build-p100/bin/llama-server --version
```

A full build takes about 15 minutes on a 6-core desktop CPU. nvcc prints
"offline compilation for architectures prior to sm_75 will be removed" for every CUDA file: that is a notice, not an error.

**If it fails:**
- `CMake was unable to find a build program corresponding to "Ninja"`: install Ninja (`ninja-build`).
- `No CMAKE_CUDA_COMPILER could be found`: `nvcc` is not on your `PATH`; add the CUDA 12.x `bin` directory, or pass
  `-DCMAKE_CUDA_COMPILER=/path/to/cuda-12.x/bin/nvcc` to the first `cmake` command.
- `Unsupported gpu architecture 'compute_60'`: that `nvcc` is CUDA 13; use CUDA 12.x.
- Link errors at the very end (`libcudart.so.12 ... not found`, `undefined reference to cudaMemcpy` / `cuMemMap`):
  the linker cannot find the CUDA libraries. Register them once:
  `echo /usr/local/cuda-12.9/lib64 | sudo tee /etc/ld.so.conf.d/cuda-12.conf && sudo ldconfig`
  (adjust the path), then run `cmake --build --preset p100` again. `libcuda.so.1` missing means the NVIDIA driver is not installed.
- If you have several CUDA versions installed, put the 12.x one first on your `PATH`.

## Recipe

Settings, launch flags and the full story: **https://youtu.be/JvLUoJYUjDM**. That launch line sets `LLAMA_SPEC_SAMPLE_TEMP=1.0`: use `0` instead (greedy drafts, faster; 0 is also this build's default when the variable is unset). Why: **https://youtube.com/watch?v=oYLZFgduLDU**.
Channel: **[Junk GPU Lab](https://www.youtube.com/@junkgpulab)**.

## Credits

- **llama.cpp / ggml** ([ggml-org](https://github.com/ggml-org/llama.cpp)), MIT, Copyright (c) 2023-2026 The ggml authors.
  The `LICENSE` file is upstream's, unchanged.
- **[Kmic-68](https://github.com/Kmic-68/llama.cpp) and co-authors (Kaden, Tyler / [Keoian](https://github.com/Keoian), kane)**, MIT:
  the sm_60 mat-vec kernels, fold-GEMM prefill, Pascal flash-attention, gated-delta-net / recurrent / norm kernels,
  compact KQ mask, fixed-width speculative verify, MTP draft vocab, server padded verify and checkpoints, tensor-parallel
  P2P exchange. All kernel design and tuning are theirs.
- **[shinbunbun](https://github.com/shinbunbun/llama-cpp-p100-patches)**, MIT: patch 01 (vmad dp4a), behind `-DGGML_CUDA_P100_VMAD` (off by default).
- Thanks to **fallentree** and **[qr0n](https://github.com/qr0n/p100-llama-cpp)** for the P100 community work (no code of theirs is in this repo).
- No model weights are shipped here; see each model card for its licence.

Full notice: [NOTICE.md](NOTICE.md).

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
