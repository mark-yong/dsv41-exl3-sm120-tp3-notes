# Candidate B: official checkpoint + decoder-half UVA expert offload

The official DeepSeek `fb2764a5` checkpoint on the LIL r38 image, with
vLLM's UVA CPU offloader re-targeted from its stock encoder-half walk to
the decoder half (layers at and above `ced_decoder_start` = 20). Recipe by
[peterkilfeather](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed)
(gist, 2026-09-21); this directory carries this box's reproduction, the
pinned overlay, a runnable compose, and the measured comparison.

## Why the decoder half

The stock offloader spends its byte budget walking layers in ascending
`make_layers` order, so `--cpu-offload-params experts` parks the first
layers' experts. For DS4.1-Flash those are the CED encoder half, which runs
over every prefill chunk row. The decoder half replays only
`min(chunk, 128)` rows per request per step, so the same 8 GiB budget spent
at ordinals 20 and up costs roughly 30x less per chunk at prefill. Decode
traffic is identical either way, which is why the same offload target is
strictly better here.

The overlay (`overlay/uva.py`, sha256
`5295291db8d6006d3c03e1dd73252a2df42ffe5c2a1469a1151a5a62dc0e8d0e`) keeps
the walk lazy, adds the ordinal predicate, and logs per-module accounting.
Because UVA keeps every offloaded parameter showing `.device == cuda` (a
CUDA view over pinned host memory), B12X plans and CUDA graphs keep
working: nothing about offload disables compile or capture. To confirm the
overlay loaded, check for the `ds4-laune overlay` line in the boot log.

I measured with DSpark OFF. Pete measured the same regression: re-enabling
DSpark under uniform offload cost 42-58 tok/s vs 75-85 with it off.

## Run it

```bash
cd candidates/uva/compose
cp .env.example .env          # set MODEL_HOST_PATH (official fb2764a5 tree)
docker compose up -d          # stop other GPU workloads first
docker logs ds41-tp3-uva 2>&1 | grep -A2 'ds4-laune overlay'   # confirm it loaded
```

Expected attribution line:

```
# ds4-laune overlay: offload walk from ordinal 20,
# (ordinal, bytes) = [(20, '2.24'), (21, '2.24'), (22, '2.24'), (23, '1.41')]
# Total CPU offloaded parameters: 8.13
```

That is layers 20-23.63 of 40, at 2.24 GiB/layer/rank. Host RAM adds ~63
GiB for the Engram tables plus 8 GiB offload on top of normal usage.

The compose is the gist original with four mechanical changes: model path
parameterised, overlay path pointed into this repo, host port 8015, project
renamed. Everything else, including the image digest, is verbatim.

## Measured (this box, PCIe 4.0 x16 NODE)

Same-box comparison vs the EXL3 candidate, 2026-09-22, exclusive GPU
window, same `llm-inference-bench` family; full tables in
[benchmarks/COMPARE.md](benchmarks/COMPARE.md):

- Decode C=1: 75.8 at 0 ctx, 75.4 at 16k, 74.8 at 32k, holding 67-72 out
  to 1M. KV pool 2.72M tokens (2.60x the 1M ceiling).
- Prefill: 4,005 at 8k (cold scout), 4,508 at 16k, 4,425 at 32k, 4,319 at
  128k (128,475 tok, TTFT 29.8 s). The 256k/512k/1M cells were decode-only
  on this engine; I did not collect standalone long prefills.
- Bring-up: first READY ~14-17 min cold (InstantTensor loading 286 GiB at
  ~1.1-1.5 GB/s via `AIO_BUFFERED,MMAP`; the graph cache cut capture from
  47 s to 13 s warm). My first two boots died on
  `io_uring_register_buffers` (unprivileged container, 8 MiB memlock)
  before the loader fell back.
- Pete's Gen5 reference (same image, flags, and offload on a PCIe Gen5
  x16/x16/x8 host): prefill 7,424 at 32k / 4,651 at 1M; decode 76.3 at 32k
  rising to 85.5 at 1M. Gen4-vs-Gen5: absolute prefill does not transfer
  1:1 (the offload reads ride PCIe); decode matches at short/mid context
  (74-76 vs 76-79) and trails at long context (67-72 vs 81-86).

## Caveats (measured by Pete, carried from the gist)

- Zero memory margin at util 0.98: ~1.9K expandable-segments retry
  warnings during the 1M prefill (rank 1). Warnings only; the engine
  stayed healthy.
- Do not re-enable DSpark under uniform offload (measured regression).
- A per-expert hot/cold split (two MoE calls per layer) collapses decode
  to 23 tok/s; the fix is a fused two-table kernel (upstream b12x work,
  not attempted here).
- Decode is wait-bound on the TP all-reduce (~40% of kernel time per
  Pete's measurement; PYNCCL ring at world-3, because the B12X PCIe AR
  rejects world size 3). Documented with the upstream links and the
  allowlist analysis in
  [b12x-410-pcie-ar-world3.md](b12x-410-pcie-ar-world3.md).
- Pete ran no fidelity gates on this mechanism beyond smoke tests (the
  weights are moved, not modified).

## Method notes for this box's run

My first bench matrix died: it warmed up at 32k first and the workers hung
in compile (`sample_tokens` RPC 900 s). Ordering the cells 0 -> 16k ->
32k -> 128k with `--decode-warmup-seconds 0` was enough. A 16k "116 tok/s"
figure from that dead first run is invalid and excluded.
