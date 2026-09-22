# DeepSeek-V4.1-Flash at TP3 on 3x RTX PRO 6000 (SM120): two candidates, one box

I wanted to see which TP3 route made more sense on my 3x RTX PRO 6000 box:
the 3.51 bpw EXL3 checkpoint, or the official DeepSeek checkpoint with
decoder-half UVA expert offload. I got both running and benchmarked them on
the same machine.

The short version: EXL3 is faster at short-context prefill. UVA is much
faster for single-user decode and is the only one of the two that I got
working at 128k-1M context. The rest of the repo is the configs,
measurements, and failed attempts behind that result.

- **Candidate A, EXL3 3.51 bpw** ([candidates/exl3/README.md](candidates/exl3/README.md)):
  the Pollard checkpoint on Jake Tempo's cuda-exl3 TP3 port.
- **Candidate B, official checkpoint + decoder-half UVA** ([candidates/uva/NOTES.md](candidates/uva/NOTES.md)):
  recipe by peterkilfeather; the UVA expert offload re-targeted to the
  decoder half.
- **Head-to-head** ([candidates/uva/benchmarks/COMPARE.md](candidates/uva/benchmarks/COMPARE.md)):
  same-box comparison of both final configs plus Pete's Gen5 numbers.

## Result

| | EXL3 3.51 bpw (P4) | official checkpoint + UVA |
|---|---|---|
| Serve | `candidates/exl3/reproduce/compose/` | `candidates/uva/compose/` |
| Port | 8014 | 8015 |
| Checkpoint | Pollard 3.51 bpw, 48 shards, 428.5 GiB, sha-pinned | official `fb2764a5` |
| Image | GHCR `dsv41f-tempo-sm120-tp3@sha256:ddd31bc7…` (build recipe in repo) | LIL r38 `localinferencelab/vllm@sha256:f41ca8bb…` |
| Max context | 32k (131k boots, OOMs under bench) | 1M window (1M decode cell passed at 1,034,830-token context) |
| KV pool | 4 GiB fp8 at 32k; a 7 GiB (reported as 3,479,680 tokens) attempt booted but OOMed under bench | 2.72M tokens realized on this box (2.60x the 1M ceiling); Pete's Gen5 host: 2.85 GiB ≈ 2.6M tokens |
| Prefill (8k / 16k / 32k) | 6,140 / 5,933 / skipped | 4,005 / 4,508 / 4,425 |
| Decode C=1 (0 / 32k / 1M) | 55.5 / skipped / — | 75.8 / 74.8 / 71.7 |
| Bring-up to READY | ~25 min | ~14–17 min on both timed boots (InstantTensor still ~4.5 min); graph capture only 47 s → 13 s warm; page-cache-hot restart including weight load **not** timed |
| Speculation | DSpark on (accept 2.3-2.4 tok/step) | DSpark off (it regressed under offload when Pete tried it) |

On my 3x RTX PRO 6000 Gen4 box, UVA is the better choice for decode and
long context; EXL3 wins short-context prefill. Pete's Gen5 host (same
image and flags) prefills 6.8-7.4k, above both of my routes, which is a
good reminder that PCIe generation changes this picture. I would not
choose EXL3 over UVA on this hardware unless I specifically needed EXL3:
if I did, the working config and image build are documented in
candidates/exl3/, so there is no need to repeat the failed bring-up
attempts.

Numbers are from `llm-inference-bench` sustained-decode runs with no other
GPU workloads running. Community figures (Pete's Gen5 gist) are cited with
their provenance, never mixed with my own measurements.

One shared bottleneck worth knowing: on this image the B12X PCIe
all-reduce rejects world size 3, so TP3 falls back to PYNCCL (roughly 40%
of decode kernel time per Pete's measurement). Upstream issue and the
acceptance path for a fix:
[b12x-410-pcie-ar-world3.md](candidates/uva/b12x-410-pcie-ar-world3.md).

## Test setup

- Hardware: 3x RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16, NODE topology;
  NVIDIA driver 615.71.09, CUDA 13.4 user mode. Pete's gist numbers come
  from a PCIe Gen5 x16/x16/x8 host; the two are labeled separately
  everywhere in this repo.
- Bench: [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  (Martin Vit) @ `d115fee` (2026-09-01); 30 s sustained decode per cell,
  2,048 max output tokens, engine-default sampling.
- The 1M cell on the UVA engine was decode-only over a 1,034,830-token
  context (TTFT 6.0 s): a context-capacity result, not a standalone 1M
  prefill or a retrieval test. Standalone 256k/512k/1M prefill was not
  collected on this engine.
- Output quality (EXL3 vs official weights) is not tested yet; what I ran
  was smoke tests and the bench tables.

## What I'd test next

1. A DSpark-off run at EXL3 P4, to isolate how much the speculation is
   actually buying (accept length was 2.3-2.4 tokens/step; I never ran the
   A/B).
2. Things I'd try to get EXL3 131k working, one change at a time:
   speculative decoding off, smaller CUDA graphs, a shorter prefill
   matrix. The last attempt missed by tens of MiB, so any one might clear
   it.
3. A ~3.25 bpw EXL3 build to free VRAM for a larger KV pool. I skipped it
   so far (I tested the existing 3.51 bpw checkpoint as-is); it is the
   first thing I'd try if EXL3 long context is the goal.
4. Standalone UVA 256k/512k/1M prefill cells (so far decode-only).

## Credits

A warm thanks to the many members of the Local Inference Lab Discord;
this repo is mostly assembled from their work.

- **Jake Tempo** ([jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek)):
  the cuda-exl3 TP3 serving runtime the EXL3 candidate is built on.
- **tonyd2wild** ([DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)):
  the upstream TP3 patch set the rebuild applies.
- **bot-lab-21**: the EXL3 3.51 bpw checkpoint (WestWaters' Pollard method,
  turboderp's EXL3 format).
- **[peterkilfeather](https://github.com/peterkilfeather)** (Pete): the
  decoder-half UVA offload overlay, compose, and the Gen5 measurements
  ([gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed))
  that candidate B reproduces.
- **[voipmonitor](https://github.com/voipmonitor)** / voipmonitor (Martin
  Vit): the docker containers
  ([blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker))
  and recipes ([rtx6kpro](https://github.com/local-inference-lab/rtx6kpro))
  most LIL projects run on, the InstantTensor loader and
  `voipmonitor/vllm` image line behind the UVA route, and the
  [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  harness every number in this repo comes from.
- **[lukealonso](https://github.com/lukealonso)** (Luke Alonso): the
  [b12x](https://github.com/local-inference-lab/b12x) kernel backend (this
  node's vLLM stacks launch with `BACKEND=b12x`), the PCIe oneshot and
  even-width all-reduce work, and the quants in
  [quant-toolkit](https://github.com/local-inference-lab/quant-toolkit).
- **[DeepSeek](https://github.com/deepseek-ai)**:
  [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
  DSpark, CED, and Engram.
- **vLLM**, **FlashInfer**, **TileLang**, **NVIDIA CUDA / SM120**.

## Licence and notices

Apache-2.0. Third-party notices preserved at the root ([NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)); per-candidate provenance
in each candidate directory. The UVA overlay is peterkilfeather's, pinned
by sha256 in candidates/uva/NOTES.md.
