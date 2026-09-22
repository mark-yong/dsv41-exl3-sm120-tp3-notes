# Known issue: B12X PCIe all-reduce rejects world size 3

Upstream issue: [local-inference-lab/b12x#410](https://github.com/local-inference-lab/b12x/issues/410)
(open, filed 2026-09-21). Related PR:
[b12x#297](https://github.com/local-inference-lab/b12x/pull/297) (world
size 3 for the one-shot/DMA all-reduce, not merged).

This affects the official-checkpoint UVA candidate on this repo, because
that route runs the LIL r38 image whose MoE path uses the B12X kernel
backend. The EXL3 candidate is unaffected (Tempo's custom all-reduce
handles world 3 with the P2P override; it does not use B12X).

## What happens

On 3-GPU TP3, the B12X PCIe all-reduce never initializes and vLLM falls
back to PYNCCL for both the `tp:0` and `ep:0` groups:

```
WARNING ... [b12x_pcie_all_reduce.py:223] B12X PCIe all-reduce
initialization failed on rank 0: unsupported PCIe all-reduce world size 3;
supported world sizes are (2, 4, 6, 8, 12, 16)
WARNING ... [custom_all_reduce.py:133] Custom allreduce is disabled due to
an unsupported world size: 3. Supported world sizes: [2, 4, 6, 8, 16].
INFO    ... [cuda_communicator.py:314] Using ['PYNCCL'] all-reduce
backends ... for group 'tp:0'
```

Pete measured the consequence on his Gen5 box (same image family, same
TP3 UVA offload): the PYNCCL TP all-reduce accounts for roughly 40% of
decode kernel time. Decode is wait-bound on the collective, which is also
what exposes the x8 card's offload reads in his setup. This is the shared
bottleneck behind the long-context decode gap in
[benchmarks/COMPARE.md](benchmarks/COMPARE.md) (67-72 here vs Pete's
81-86 at 256k-1M; part of that gap is Gen4-vs-Gen5, and this fallback is
the other part).

The image digest this repro used:
`localinferencelab/vllm@sha256:f41ca8bb10bb3a125a50340d70d39ad4b7f5605f3fcb661bc992ed0bc4701a00`
(`jovian-judgement-community-20260914-r38`).

## The allowlist discrepancy

The r38 wrapper advertises one supported-world-size set and current
`master` another:

- r38 image (`b12x_pcie_all_reduce.py`): `(2, 4, 6, 8, 12, 16)`
- [master `pcie_oneshot.py`](https://github.com/local-inference-lab/b12x/blob/master/b12x/comm/pcie/pcie_oneshot.py):
  `(2, 4, 6, 8, 10)`

Neither includes 3, and the sets are not monotonically stricter of each
other. Meanwhile the generic CuTe pull path does not look even-N-specific
in code: its barrier operates over `self._world_size` and its reduction
walks every peer with `range_constexpr(self._world_size)` and modulo rank
selection. The explicitly topology-specific restrictions are TP2
remote-push, TP4 fused remote-push, and TP8 owner-reduce, not the generic
pull algorithm. So TP3 looks like an allowlist, specialization, and
qualification gap rather than a new collective, though a compile-cache,
IPC-layout, planner, or graph-replay assumption could still hide in the
staging/barrier/slot-management path. The torture test itself hardcodes
the same even allowlist, so world size 3 has never passed real
qualification.

## What would change it

[b12x#297](https://github.com/local-inference-lab/b12x/pull/297) already
adds `3` to the supported sets in `pcie_oneshot.py` and `pcie_dma.py`
(opened 2026-09-02, not merged), but does not replace the issue as filed:
it carries a GLM-5.3-Flash policy corpus rather than a DS4.1
serving-path pin, has no WS3 entry in the torture suite, and no FP32-sum
oracle or CUDA-graph capture/replay gate at world 3.

The acceptance path from the issue, before any timing claim:

1. Eager oneshot at world 3 for BF16 DS4.1 decode shapes, with
   rank-adversarial inputs and an FP32-sum oracle, plus a separate NCCL
   comparison.
2. The same shapes under CUDA graph capture and replay.
3. The standard oneshot test and GPU torture suite accepting 3 the way
   they accept 2/4/6/8/10.
4. Only then, autotune crossover vs PYNCCL/NCCL on 3x SM120 and a real
   TP3 decode cell.

I offered to re-bench the r38 DS4.1 TP3 UVA C=1 path on this box once a
pin lands. If it does, the decode rows in
[benchmarks/COMPARE.md](benchmarks/COMPARE.md) get re-measured and this
file records the delta.
