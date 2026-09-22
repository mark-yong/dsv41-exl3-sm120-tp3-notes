# Same-box A/B — dense TP3 UVA vs EXL3 P4 vs Pete Gen5

Measured 2026-09-22 on the **same 3× RTX PRO 6000** box as the EXL3 notes
in [README.md](README.md): PCIe **Gen4** x16 NODE. Exclusive GPU window.
Same `llm-inference-bench` family as the P4 tables.

**This box, dense UVA:** LIL r38 `localinferencelab/vllm@sha256:f41ca8bb…`,
DSpark **off**, decoder-half UVA 8.13 GiB (ordinals 20–23), Engram RAM,
util 0.98, batched 2048, capture 24, `INSTANTTENSOR_BACKEND=AIO_BUFFERED,MMAP`.
Recipe: [peterkilfeather gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed).

**Pete:** same image/flags/offload; PCIe **Gen5** x16/x16/x8 Max-Q. Gist
2026-09-17.

**EXL3 P4 keep:** Pollard 3.51 Tempo SM120, DSpark **on**, batched 4096,
Engram **pinned DDR**, 32k envelope; 131k failed. Details in the README.

## Wall clock (same box)

Fair compare is **keep-config bring-up**, not the EXL3 P0–P6 climb.

| | EXL3 P4 keep | This box UVA (live engine) |
|---|---|---|
| `model_runner` load | **1035 s (17.3 min)**, 83.18 GiB/rank | **693 s (11.6 min)**, 87.53 GiB/rank |
| Weight I/O | `"Loading weights took 96.73 s"` after DSpark draft | InstantTensor **286 GiB in 4:33** @ ~1.1–1.5 GB/s (`AIO_BUFFERED,MMAP`) |
| CUDA graphs | TileLang/FlashInfer JIT after load (~6 min to routes) | First READY **47 s + 8 s**; live engine (cache hit) **13 s + 8 s** |
| Start → `Application startup complete` | **~25 min** | First AIO READY **~17 min**; this engine **~14 min** |

Those 14–17 min / 25 min rows are **cold-ish first READY**, not a warm
restart. Graph/JIT cache already cut UVA capture **47 s → 13 s** between
the first successful boot and the later recreate. InstantTensor on that
recreate was still ~**1.1–1.5 GB/s** (NVMe-like), so the 286 GiB read did
not yet come from DRAM.

A later restart with Linux page cache hot (this host has ~500 GiB RAM;
286 GiB weights + ~63 GiB Engram can stay resident) would drop most of
that **4:33** I/O. That restart was not timed — the live engine stayed up
through 1M. EXL3 does **not** show the same load-time win across rungs:
P1 `model_runner` was **630 s**, P4 keep **1035 s** (Engram pin + DSpark
on top of JIT cache).

First two UVA boots aborted on `io_uring_register_buffers` (unprivileged
container, 8 MiB memlock) before `AIO_BUFFERED,MMAP`.

## Prefill (tok/s)

| Context | This box UVA | EXL3 P4 keep | Pete UVA (Gen5) |
|---------|--------------|--------------|-----------------|
| 8k | 4005 (cold scout, first boot) | **6140** | — |
| 16k | **4508** | **5933** (server 6168) | — |
| 32k | **4425** | skipped | **7424** |
| 128k | **4319** (128475 tok, TTFT 29.8s) | **failed** (131k) | **6806** |
| 256k | *not collected* (decode-only cell) | — | **6331** |
| 512k | *not collected* (decode-only cell) | — | **5578** |
| 1M | *not collected* (decode-only cell) | — | **4651** |

## Decode C=1 (tok/s)

| Context | This box UVA | EXL3 P4 | Pete UVA (DSpark off) |
|---------|--------------|---------|------------------------|
| 0 | **75.8** | ~55.5 | ~76–85 band |
| 16k | **75.4** | ~56.0 | — |
| 32k | **74.8** | skipped | **76.3** |
| 128k | **73.8** | — | **79.0** |
| 256k | **71.8** | — | **80.6** |
| 512k | **67.1** | — | **84.1** |
| 1M | **71.7** (`1034830` tok, TTFT 6.0s) | — | **85.5** |

256k / 512k / 1M were **decode-only**. TTFTs 1.6s / 3.1s / 6.0s are not
standalone prefills (a 1M prefill at ~4.3k tok/s would be minutes; Pete’s
1M TTFT was 222s).

## Read

- **Decode matches Pete at short/mid context** (74–76 vs 76–79). Long-context
  C=1 holds 67–72 vs Pete 81–86. EXL3 P4 ~55 is beaten by ~35%. DSpark
  stayed off on UVA.
- **Prefill does not.** ~4.3–4.5k vs Pete 6.8–7.4k at 32k/128k, and below
  EXL3 P4 6.1k@8k. Mix: Gen4 vs Gen5, UVA decoder-expert PCIe reads,
  batched 2048 vs P4’s 4096. PYNCCL at TP3 is **shared** with Pete (B12X
  PCIe AR rejects world size 3; [b12x#410](https://github.com/local-inference-lab/b12x/issues/410),
  related [b12x#297](https://github.com/local-inference-lab/b12x/pull/297)).
- **128k and 1M decode work.** That is the EXL3 Path A′ hole. KV pool
  2.72M tokens (2.60× at 1M).
- **Bring-up:** first UVA READY ~14–17 min vs EXL3 P4 ~25 min. Warm graph
  cache already 47 s → 13 s; a page-cache-hot UVA restart was not timed.

## Method notes

First matrix died: bench warmed up at 32k first; workers hung in compile
(`sample_tokens` RPC 900s). Ordered ctx=0 → 16k → 32k → 128k with
`--decode-warmup-seconds 0` was enough. A 16k “116 tok/s” figure from the
dead first run is invalid.

InstantTensor `URING` / `BUFFERED` abort on this unprivileged container
(8 MiB memlock); `AIO_BUFFERED,MMAP` is the working loader.

No standalone 256k / 512k / 1M prefill on this engine.

## Credits

This run is a homelab reproduction. The recipe, image, kernels, and weights
are not ours.

- **[Pete / peterkilfeather](https://github.com/peterkilfeather)** —
  decoder-half UVA overlay, compose, and Gen5 ladder
  ([gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed)).
- **[Local Inference Lab](https://github.com/local-inference-lab)** — r38
  [`localinferencelab/vllm`](https://github.com/local-inference-lab/vllm)
  (`sha256:f41ca8bb…`), [`b12x`](https://github.com/local-inference-lab/b12x),
  [`rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro),
  [`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
  (Martin Vit).
- **[Voipmonitor](https://github.com/voipmonitor)** / voipmonitor —
  InstantTensor loader, the `voipmonitor/vllm` image line, and the earlier
  [`rtx6kpro`](https://github.com/voipmonitor/rtx6kpro) PCIe/Docker notes.
- **[Luke Alonso](https://github.com/lukealonso)** — B12X PCIe oneshot,
  even-width AR, MiniMax TP3 virtual-shard pattern (`lukealonso/sglang`).
- **[DeepSeek](https://github.com/deepseek-ai)** —
  [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  `fb2764a5`, CED, Engram.
- **vLLM**, **FlashInfer**, **TileLang**, **NVIDIA CUDA / SM120**.
- **EXL3 Path A′ compare:**
  [bot-lab-21 Pollard](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard),
  [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
  TP3 overlay, Tempo / cuda-exl3.
