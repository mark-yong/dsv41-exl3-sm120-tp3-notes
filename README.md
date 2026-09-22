# DeepSeek-V4.1-Flash at TP3 on 3x RTX PRO 6000 (SM120): two candidates, one box

Two ways to serve DeepSeek-V4.1-Flash at tensor parallelism 3 on three RTX
PRO 6000 96 GB cards (Blackwell SM120, PCIe 4.0 x16 NODE, no NVLink), built
and measured on the same box, with a head-to-head of the measured keep
configs:

- **Candidate A, EXL3 3.51 bpw** ([candidates/exl3/README.md](candidates/exl3/README.md)):
  the Pollard checkpoint on Jake Tempo's cuda-exl3 TP3 port. Working but not
  winning: 6.1k prefill @ 8k, ~55 tok/s per-user decode (DSpark on), 32k
  max context; 131k OOMs under bench. Full bring-up ladder, failure
  anatomy, and image build.
- **Candidate B, dense + decoder-half UVA** ([candidates/uva/NOTES.md](candidates/uva/NOTES.md)):
  the dense checkpoint with vLLM's UVA expert offload re-targeted to the
  decoder half, recipe by peterkilfeather. 74-76 tok/s per-user decode,
  serving 128k and 1M context (KV pool 2.72M tokens); prefill ~4.3-4.5k on
  this Gen4 box, below Pete's Gen5 6.8-7.4k reference.
- **Head-to-head** ([candidates/uva/receipts/COMPARE.md](candidates/uva/receipts/COMPARE.md)):
  same-box A/B of both keep configs plus Pete's Gen5 numbers.

## The one-paragraph answer

Long context and per-user decode: dense+UVA is the working path on this
hardware class, and it is not close (75 vs 55 tok/s C=1; 1M works vs 131k
OOM). Short-context prefill: EXL3 wins (6.1k vs 4.5k @ 16k). If you need
EXL3 specifically, the working config and both failure modes are documented
in candidates/exl3/, so the bring-up cost is the build, not the debugging.

## Quick reference

| | EXL3 3.51 bpw (P4) | dense + UVA |
|---|---|---|
| Serve | `candidates/exl3/reproduce/compose/` | `candidates/uva/compose/` |
| Port | 8014 | 8015 |
| Checkpoint | Pollard 3.51 bpw, 48 shards, 428.5 GiB, sha-pinned | dense `fb2764a5` |
| Image | GHCR `dsv41-tempo-sm120-tp3@sha256:ddd31bc7…` (build recipe in repo) | LIL r38 `localinferencelab/vllm@sha256:f41ca8bb…` |
| Max context | 32k (131k boots, OOMs under bench) | 1M (1,034,830-token needle PASS) |
| Prefill (8k / 16k / 32k) | 6,140 / 5,933 / skipped | 4,005 / 4,508 / 4,425 |
| Decode C=1 (0 / 32k / 1M) | 55.5 / skipped / — | 75.8 / 74.8 / 71.7 |
| Bring-up to READY | ~25 min | ~14-17 min cold; capture 13 s warm |
| Speculation | DSpark on (accept 2.3-2.4 tok/step) | DSpark off (regresses under offload) |

Numbers are from `llm-inference-bench` sustained-decode runs during an
exclusive GPU window; per-cell data and methodology in the candidate
directories. Community figures (Pete's Gen5 gist) are cited as external
inputs, never as results from this box.

## Shared facts

- Hardware: 3x RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16, NODE topology;
  NVIDIA driver 615.71.09, CUDA 13.4 user mode. Pete's gist numbers come
  from a PCIe Gen5 x16/x16/x8 host: the two are labeled separately
  everywhere in this repo.
- Bench: [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  (Martin Vit) @ `d115fee` (2026-09-01); 30 s sustained decode per cell,
  2,048 max output tokens, engine-default sampling.
- The 1M decode cell on the UVA engine passed with a 1,034,830-token prompt
  (TTFT 6.0 s, decode-only cell); standalone 256k/512k/1M prefill was not
  collected on this engine.
- Both engines ran in evaluation tenancy: no other GPU tenants during
  measurement windows.

## What is next

Untried, in rough priority order:

1. DSpark-off decode rung at EXL3 P4: isolates the net wall-clock speedup
   of speculation (accept length 2.3-2.4 tokens/step; the A/B was not run).
2. EXL3 long-context recoveries, one knob at a time (speculative decoding
   off, smaller CUDA graphs, shorter prefill matrix): the 131k bench fell
   short by tens of MiB, so any one may clear it.
3. A ~3.25 bpw auto-build to free VRAM for a larger EXL3 KV pool: the
   obvious lever if EXL3 long context is the goal (so far out of scope by
   the policy note in candidates/exl3/).
4. Standalone UVA 256k/512k/1M prefill cells (decode-only so far).
5. A real fidelity check: current validation is smoke tests and bench
   tables only; nothing measures output quality against dense.

## Licence and notices

Apache-2.0. Third-party notices preserved at the root ([NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)); per-candidate provenance
in each candidate directory. The UVA overlay is peterkilfeather's, pinned
by sha256 in candidates/uva/NOTES.md.
