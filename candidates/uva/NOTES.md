# Candidate B: dense weights + decoder-half UVA expert offload

Dense `fb2764a5` checkpoint on the LIL r38 image, with vLLM's UVA CPU
offloader re-targeted from the stock encoder-half walk to the decoder half
(layers ≥ `ced_decoder_start` = 20). Recipe by
[peterkilfeather](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed)
(gist, 2026-09-21); this directory carries this box's reproduction, the
pinned overlay, a runnable compose, and the measured comparison.

## The mechanism (from the gist, verified against the pinned overlay source)

The stock offloader spends its byte budget walking layers in ascending
`make_layers` order, so `--cpu-offload-params experts` parks the first
layers' experts: for DS4.1-Flash that is the CED encoder half, which runs
over every prefill chunk row. The decoder half replays only
`min(chunk, 128)` rows per request per step, so the same 8 GiB budget parked
at ordinals ≥ 20 is ~30x cheaper per chunk at prefill. Decode traffic is
identical either way.

The overlay (`overlay/uva.py`, sha256
`5295291db8d6006d3c03e1dd73252a2df42ffe5c2a1469a1151a5a62dc0e8d0e`) keeps
the walk lazy, adds the ordinal predicate, and logs per-module accounting:
the `ds4-laune overlay` boot line is the attribution proof, not inference.
UVA keeps every offloaded parameter's `.device == cuda` (a CUDA view over
pinned host memory), so B12X plans and CUDA graphs stay valid. Numbers below
measured with DSpark OFF (re-enabling it under uniform offload measured
42-58 tok/s vs 75-85 with it off, per the gist's caveats).

## Reproduce on this box

```bash
cd candidates/uva/compose
cp .env.example .env          # set MODEL_HOST_PATH (dense fb2764a5 tree)
docker compose up -d          # evaluation tenancy: stop other GPU tenants first
docker logs ds41-tp3-uva 2>&1 | grep -A2 'ds4-laune overlay'   # attribution proof
```

The compose is the gist original with four mechanical deltas (model path
parameterised, overlay path pointed into this repo, host port 8015, project
renamed); everything else, including the image digest, is verbatim.

## Measured (this box, PCIe 4.0 x16 NODE; full tables in receipts/COMPARE.md)

Same-box A/B vs the EXL3 candidate, 2026-09-22, exclusive GPU window, same
`llm-inference-bench` family:

- Decode C=1: 75.8 @ 0 ctx, 75.4 @ 16k, 74.8 @ 32k, holding 67-72 out to 1M.
  KV pool 2.72M tokens (2.60x the 1M ceiling).
- Prefill: 4,005 @ 8k (cold scout), 4,508 @ 16k, 4,425 @ 32k, 4,319 @ 128k
  (128,475 tok, TTFT 29.8 s). 256k/512k/1M cells were decode-only on this
  engine; standalone long prefills not collected.
- Bring-up: first READY ~14-17 min cold (InstantTensor 286 GiB @ ~1.1-1.5
  GB/s via `AIO_BUFFERED,MMAP`; graph cache cut capture 47 s -> 13 s warm).
  First two boots died on `io_uring_register_buffers` (unprivileged
  container, 8 MiB memlock) before the loader fell back.
- Pete's Gen5 reference (same image/flags/offload on PCIe Gen5 x16/x16/x8):
  prefill 7,424 @ 32k / 4,651 @ 1M; decode 76.3 @ 32k rising to 85.5 @ 1M.
  Gen4-vs-Gen5: absolute prefill does not transfer 1:1 (the offload reads
  ride PCIe); decode matches at short/mid context (74-76 vs 76-79) and
  trails long context (67-72 vs 81-86).

## Known caveats (carried from the gist, all measured there)

- Zero memory margin at util 0.98: ~1.9K expandable-segments retry warnings
  during the 1M prefill (rank 1). Warnings only; engine healthy.
- Do not re-enable DSpark under uniform offload (measured regression).
- Per-expert hot/cold split collapses decode to 23 tok/s; the fix is a fused
  two-table kernel (upstream b12x work, not attempted here).
- Decode is wait-bound on the TP all-reduce (~40% of kernel time; PYNCCL
  ring at world-3; the b12x PCIe AR rejects world size 3, b12x#410).
- No fidelity gates on this mechanism beyond smoke tests (weights moved,
  not modified).

## Method notes for this box's run

First bench matrix died (warm-up at 32k first; workers hung in compile,
`sample_tokens` RPC 900 s). Ordered ctx 0 -> 16k -> 32k -> 128k with
`--decode-warmup-seconds 0` was enough. A 16k "116 tok/s" figure from the
dead first run is invalid and excluded.
