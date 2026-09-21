# DeepSeek-V4.1-Flash EXL3 TP3 on 3× RTX PRO 6000 (SM120)

This repo records a bring-up of the Pollard 3.51 bpw EXL3 checkpoint of
DeepSeek-V4.1-Flash on 3× RTX PRO 6000 96 GB (Blackwell SM120) with tensor
parallelism 3, plus the performance climb that followed. It covers the config
that measured best, the prefill and decode numbers behind it, and the two
attempts that failed (FlashInfer autotune, 131k context). Successive configs
are labeled P0 through P6. The full runbook, compose files, and patch tree
live in my private homelab docs; this page is the public summary.

## Stack

| Piece | Choice |
|-------|--------|
| Checkpoint | [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) (local TP3 72-head / 9 `o_groups` tree) |
| Runtime | Jake Tempo's cuda-exl3 TP3 (Spark overlays), rebuilt for amd64 with `ARCH_LIST=12.0a` |
| Upstream TP3 patches | [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), `patch/exl3-tp3/` |
| Rebuild reference | [jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek) |
| Day-0 TP2 recipe | [diffbot EXL3 2.0bpw](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000), used for the first SM120 TP2 attempt; the working TP3 path is the Tempo rebuild |

Policy: the 3.51 bpw checkpoint was measured first, and no ~3.25 bpw
auto-build was attempted.

## What worked

The keep config is P4, measured at 32k context with a 4 GiB KV pool:

- Engram tables in pinned host DDR (`DSV41_ENGRAM_DISK=0`); the P0 default
  offloads them to disk
- Custom all-reduce on
- `NCCL_P2P_DISABLE=0`, so P2P is enabled on this SM120 box
- `max-num-seqs=4`, `max-num-batched-tokens=4096`
- FlashInfer autotune, JIT, and CuteDSL warmup off
- Vision and DSpark speculative decoding on (`num_speculative_tokens=5`),
  same as the rest of the climb

### Prefill (tok/s)

| Step | Change | 8k | 16k |
|------|--------|----|-----|
| P0 | baseline: Engram on disk, all-reduce off, P2P off | ~2.2k | not measured |
| P1 | batched 4096, seqs 2 | 2407 | 2491 |
| P2 | seqs 4 | 2353 | 2457 |
| P3 | custom all-reduce + P2P | 4137 | 4393 |
| P4 | Engram to pinned DDR | 6140 | 5933 |

dense + ordinal UVA on similar 3×96 GB hardware measured 7,424 prefill
tok/s at 32k and 4,651 at 1M; the reference is [peterkilfeather's
decoder-half UVA offload gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed),
a vLLM overlay that parks 8.1 GiB per rank of decoder-half routed experts
(layers 20 and up) in pinned host RAM, DSpark off. The P4 peak here is
~6.1k at 8k and ~5.9k at 16k, and no long-context prefill completed. P4's
gain is measured against this stack's own P0 baseline; the dense path is
faster at every point that was measured.

### Decode (P4, DSpark on, `llm-inference-bench` sustained)

peterkilfeather's gist measures decode at 76.3 tok/s per user at 32k up to
85.5 at 1M with speculative decoding off. The table is aggregate completion
tok/s; the conc-4 column is four users sharing the system.

| ctx \ conc | 1 | 2 | 4 |
|-------------|---|---|---|
| 0 | 55.5 | 108.2 | 200.9 |
| 16k | 56.0 | 107.5 | 214.2 |

- Per-request decode is about 50-56 tok/s.
- DSpark accept length measured 2.3-2.4, so MTP-normalized engine steps run
  about 23 per second at conc 1 (completion tok/s divided by accept length).
  Speculation helps; the level still trails dense + ordinal offload.
- The 32k decode cells error in the bench matrix when prompt plus 2048
  output tokens crosses the 32768 `max_model_len`; the server itself stayed
  up.

## What failed

### P5: FlashInfer autotune

Cold mxfp8 autotune ran about 61 minutes, then the TP2 rank died at the
autotune `world.barrier()` (Gloo: connection closed by peer). The engine
never reached `Application startup complete`. I dropped it rather than
re-running it for the context climb.

### P6: 131k context

| Attempt | Result |
|---------|--------|
| 8 GiB KV, util 0.90, then a leaner seqs 2 / batched 2048 retry | Boot OOM both times; ~7.6-7.9 GiB free against the 8 GiB pool |
| 7 GiB KV, seqs 2, batched 2048 | Engine ready and smoke passed; KV budget reported as 3,479,680 tokens |
| Same config, long prefill bench | CUDA OOM mid-bench; 474 MiB allocation against 417-457 MiB free |

The idle pool fits, but activation memory, CUDA graphs, speculative
decoding, and long-prefill workspace together exceed the remaining margin
under load. 131k never passed under bench. P7 (~300k) was not attempted.
Untried recoveries from the notes: turn speculative decoding off, bench a
shorter prefill matrix first, or free more VRAM with smaller graphs.

## Takeaways

1. EXL3 TP3 boots on SM120 only after a Tempo/cuda-exl3 port; the day-0
   TP2 recipe mounts alone were not the working path.
2. Within EXL3, the prefill levers that measured were custom all-reduce
   plus P2P (2.2k to 4.1k at 8k) and then Engram in pinned DDR (4.1k to
   6.1k). Batch size and seq limits moved little.
3. P4 recovers EXL3 against its own earlier baseline only. Prefill around
   6k at 8-16k and decode around 55 tok/s per user with DSpark on both
   trail the dense path (7,424 prefill at 32k, 76.3-85.5 decode).
4. For long context on this hardware class, dense weights with ordinal UVA
   expert offload (decoder-half experts parked, CED boundary at layer 20;
   see peterkilfeather's gist) measured far better than pushing the EXL3
   KV pool.
5. Engram holds native table weights rather than EXL3 experts; reuse across
   serve images only works for the same Flash revision, and different HF
   cuts need a config match check.
6. On Docker's containerd snapshotter, `docker save` and a naive
   `ctr images export` can produce empty or broken archives. Plan image
   archival with that in mind.

## Not claimed

- Fidelity checks were smoke tests and the bench tables above; nothing else
  was run.
- Nothing here claims 300k or 1M context on EXL3.
- Homelab wiring, compose files, and secrets stay in the private repo.

## Related reading

- [peterkilfeather's decoder-half UVA offload gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed):
  the dense-weights path this page compares against; vLLM UVA overlay and
  compose, measured from 32k through 1M
- Jake Tempo / Spark TP3 EXL3 lineage (links in the table above)

---

*Recorded 2026-09-21. Hardware: 3× RTX PRO 6000 96 GB SM120, PCIe. Numbers
from `llm-inference-bench` during an exclusive GPU window.*
