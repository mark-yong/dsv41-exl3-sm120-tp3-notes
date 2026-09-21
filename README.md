# DeepSeek-V4.1-Flash EXL3 TP3 on 3× RTX PRO 6000 (SM120) — experiment notes

Short public record of a Path A′ bring-up and performance climb on **3× RTX PRO 6000 96 GB (Blackwell SM120), TP3**, using the released **Pollard 3.51 bpw EXL3** checkpoint.

Not a full recipe dump. Goal: save others the dead ends.

## Stack (pins)

| Piece | Choice |
|-------|--------|
| Checkpoint | [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) (+ local TP3 72-head / 9 `o_groups` tree) |
| Runtime lineage | Jake Tempo / cuda-exl3 TP3 (Spark overlays), rebuilt for **amd64 + `ARCH_LIST=12.0a`** |
| Upstream TP3 patches | [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) `patch/exl3-tp3/` |
| Tempo rebuild reference | [jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek) |
| Day-0 SM120 EXL3 (TP2) | [diffbot EXL3 2.0bpw recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000) — used earlier; **not** the working TP3 path |

**Policy we followed:** measure 3.51 bpw first; do **not** auto-build ~3.25 bpw.

## What worked (keep config)

**P4 — best measured daily EXL3 profile @ 32k / 4 GiB KV**

- Engram in **pinned host DDR** (not disk)
- Custom all-reduce **on**
- `NCCL_P2P_DISABLE=0` (P2P enabled) on this SM120 box
- `max-num-seqs=4`, `max-num-batched-tokens=4096`
- FlashInfer autotune / JIT / CuteDSL warmup **off**
- Vision + **DSpark on** (`num_speculative_tokens=5`) — same as climb baseline

### Prefill ladder (tok/s)

| Step | Change | 8k | 16k |
|------|--------|----|-----|
| P0 | Baseline (Engram disk, AR off, P2P off) | ~2.2k | — |
| P1 | batched 4096, seqs 2 | 2407 | 2491 |
| P2 | seqs 4 | 2353 | 2457 |
| P3 | custom AR + P2P | **4137** | **4393** |
| **P4** | **Engram → pinned DDR** | **6140** | **5933** |

### Decode @ P4 (with DSpark) — `llm-inference-bench` sustained

These look modest vs dense+UVA (~75–85 tok/s/user on similar hardware with DSpark **off**). They are completion tok/s, not a missing 10×.

| ctx \ conc | 1 | 2 | 4 |
|-------------|---|---|---|
| 0 | **55.5** | 108.2 | 200.9 |
| 16k | **56.0** | 107.5 | 214.2 |

- **Per-request** ≈ 50–56 tok/s (conc4 aggregate is system throughput ≈ 4× that).
- DSpark **accept length ≈ 2.3–2.4** → MTP-normalized **engine steps/s ≈ 23** @ conc1 (tok/s ÷ accept_len). Speculation helps, but EXL3 TP3 decode here is still far from dense+ordinal-offload.

## What failed

### P5 — FlashInfer autotune

Cold mxfp8 autotune ran ~**61 minutes**, then **TP2 died** at the autotune `world.barrier()` (Gloo: connection closed by peer). Never reached `Application startup complete`. Not worth re-running for a context climb.

### P6 — 131k context

| Attempt | Result |
|---------|--------|
| 8 GiB KV / high util | **Boot OOM** (~7.6–7.9 GiB free vs 8 GiB pool) |
| 7 GiB KV, lean seqs=2 / batched=2048 | **READY** + smoke OK; engine reported ~3.48M token KV budget |
| Same, long prefill bench | **Mid-bench CUDA OOM** (~474 MiB alloc with ~417–457 MiB free) |

Idle pool fits; activation / graphs / speculative / long-prefill workspace does not. **No PASS** at 131k under load. P7 (~300k) not attempted.

## Takeaways

1. **EXL3 TP3 on SM120 is real** if you port Tempo/cuda-exl3 (not day-0 TP2 mounts alone).
2. **Prefill levers that mattered:** custom AR + P2P, then Engram pinned DDR. Batch/seqs alone were small.
3. **EXL3 wins on prefill knobs @ 32k** (P4); **decode stays ~55 tok/s/user even with DSpark** — useful as a local driver, not competitive with dense+UVA decode. Not a path to 131k–1M with DSpark/graphs still on.
4. For long context on this hardware class, **dense + ordinal UVA expert offload** (park **decoder-half** experts, CED boundary ~layer 20 — see LIL / pete8359 writeups) measured far better than pushing EXL3 KV.
5. Engram is native table weights (not EXL3 experts). Reuse across serve images only for the **same Flash revision**; different HF cuts need a config match check.
6. On Docker **containerd snapshotter**, `docker save` / naive `ctr images export` may produce empty/broken archives — plan image archival accordingly.

## Non-goals / not claimed

- No fidelity gates beyond smoke + llm-inference-bench tables above
- No claim of 300k/1M on EXL3
- Homelab wiring, compose, and secrets stay private

## Related reading

- Jake Tempo / Spark TP3 EXL3 lineage (links above)
- Local Inference Lab dense DS4.1 TP3 + UVA offload campaigns (ordinal retarget vs stock ascending offload)

---

*Recorded 2026-09-21. Hardware: 3× RTX PRO 6000 96 GB SM120, PCIe. Numbers from `llm-inference-bench` during an exclusive GPU window.*
