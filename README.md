# DeepSeek-V4.1-Flash EXL3 TP3 on 3× RTX PRO 6000 (SM120)

This repo records a bring-up of the Pollard 3.51 bpw EXL3 checkpoint of
DeepSeek-V4.1-Flash on 3× RTX PRO 6000 96 GB (Blackwell SM120) with tensor
parallelism 3, plus the performance climb that followed. It covers the config
that measured best, the prefill and decode numbers behind it, and the two
attempts that failed (FlashInfer autotune, 131k context). Successive configs
are labeled P0 through P6. The `reproduce/` directory has the image build,
compose file, and helper scripts to run it; runbooks, ops logs, and secrets
stay in my private homelab docs.

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

The keep config is P4, serving with a 32,768-token maximum context and a
4 GiB KV pool; the reported measurements are at 16k context and below:

- Engram tables in pinned host DDR (`DSV41_ENGRAM_DISK=0`); the P0 default
  offloads them to disk
- Custom all-reduce on
- `NCCL_P2P_DISABLE=0`, so P2P is enabled on this SM120 box
- `max-num-seqs=4`, `max-num-batched-tokens=4096`
- CUDA graphs on (PIECEWISE); only P0 ran eager
- FlashInfer autotune, JIT, and CuteDSL warmup off
- Vision and DSpark speculative decoding on (`num_speculative_tokens=5`),
  same as the rest of the climb

### Prefill (tok/s)

| Step | Change | 8k | 16k |
|------|--------|----|-----|
| P0 | baseline: Engram on disk, all-reduce off, P2P off, eager | ~2.2k | not measured |
| P1 | graphs, vision, DSpark on; batched 2048 to 4096, seqs 1 to 2 | 2407 | 2491 |
| P2 | seqs 4 | 2353 | 2457 |
| P3 | custom all-reduce + P2P | 4137 | 4393 |
| P4 | Engram to pinned DDR | 6140 | 5933 |

Exact control state per rung: P0 was text-only and eager (no CUDA graphs,
no vision, no DSpark); P1 turned graphs, vision, and DSpark on together
and changed the scheduler knobs in the same step; P2, P3, and P4 changed
only the knob listed. So the P0-to-P1 jump bundles graph mode, vision,
speculation, and scheduler changes, and the P1-to-P4 climbs are
single-knob. The compose file's header comments carry the same mapping.

dense + ordinal UVA on similar 3×96 GB hardware measured 7,424 prefill
tok/s at 32k and 4,651 at 1M; the reference is [peterkilfeather's
decoder-half UVA offload gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed),
a vLLM overlay that parks 8.1 GiB per rank of decoder-half routed experts
(layers 20 and up) in pinned host RAM, DSpark off. The P4 peak here is
~6.1k at 8k and ~5.9k at 16k, and no long-context prefill completed. P4's
gain is measured against this stack's own P0 baseline. The dense/UVA
reference reports higher throughput under a different serving
configuration; the two were not A/B-tested here.

### Decode (P4, DSpark on, `llm-inference-bench` sustained)

peterkilfeather's gist measures decode at 76.3 tok/s per user at 32k up to
85.5 at 1M with speculative decoding off, under a different serving
configuration; not an A/B against the table. The table is aggregate
completion tok/s; the conc-4 column is four users sharing the system.

| ctx \ conc | 1 | 2 | 4 |
|-------------|---|---|---|
| 0 | 55.5 | 108.2 | 200.9 |
| 16k | 56.0 | 107.5 | 214.2 |

- Per-request decode is about 50-56 tok/s.
- DSpark accept length measured 2.3-2.4 tokens per step (MTP-normalized
  engine steps about 23 per second at conc 1). A DSpark-off A/B was not
  run, so the net wall-clock speedup from speculation was not isolated.
- The 32k decode cells error in the bench matrix when prompt plus 2048
  output tokens crosses the 32768 `max_model_len`; the server itself stayed
  up.

## What failed

### P5: FlashInfer autotune

Cold mxfp8 autotune ran about 61 minutes, then tensor-parallel rank 2 died
at the autotune `world.barrier()` (Gloo: connection closed by peer). The
engine never reached `Application startup complete`. I dropped it rather
than re-running it for the context climb.

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

## Reproducing

The `reproduce/` directory carries the working config and scripts. It
assumes a 3× RTX PRO 6000 (Blackwell SM120) host with recent NVIDIA driver
(615.71.09 and CUDA 13.4 user mode in this run) and Docker with the CDI
NVIDIA runtime.

1. Weights. Pull
   [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)
   at revision
   [`f129e31a81e1337aa33e129e2d847fc7e37c8733`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard/tree/f129e31a81e1337aa33e129e2d847fc7e37c8733)
   (48 shards, 428.5 GiB), verify it, then build the TP3 hardlink tree
   with the virtual-heads config:

   ```bash
   # name the local dir after the revision; hf CLI equivalent:
   # hf download bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard \
   #   --revision f129e31a81e1337aa33e129e2d847fc7e37c8733 --local-dir <dir>
   python3 reproduce/check-model-manifest.py <dir>   # sizes + SHA-256 vs manifest
   bash reproduce/tools/prep-tp3-config.sh           # SRC=<dir>
   ```

   `reproduce/model-manifest.json` pins every file of the revision
   (filename, byte size, and LFS SHA-256; 51 hashed files including all
   48 shards). The checker streams hashes and fails on any mismatch.
   The prep script re-checks shard presence, hardlinks the files into a
   `-TP3` tree (no extra disk), and rewrites `config.json` to 72 heads /
   9 `o_groups` with the original recorded in `virtual_heads_from`.

   The checkpoint ships Python files, and the server runs with
   `--trust-remote-code`, so this revision pin plus the manifest check is
   the supply-chain gate; do not skip it on an untrusted network path.

   On revision identity: Tempo's third-party notices record this
   checkpoint's card snapshot as
   `b60193e0609147553145d1538d935925f2763c1d`. The two revisions' trees
   are byte-identical except the HF card `README.md` (compared via the HF
   API on 2026-09-21), so either snapshot yields the same weights and
   configs. This repo pins `f129e31a…` and that is what was measured; the
   manifest check is the byte-level gate either way.

2. Host P2P override. The P3 step (custom all-reduce + P2P) ran with PCIe
   P2P forced for the NODE topology. **Only do this if your topology
   matches and you understand what ForceP2P changes** — it forces the
   driver to expose P2P over paths that may not truly support it, which
   is exactly what this NODE/PCIe layout needed but other layouts may
   not:

   ```bash
   sudo cp reproduce/host/nvidia-p2p-override.conf /etc/modprobe.d/
   sudo update-initramfs -u && sudo reboot
   ```

   Verify after reboot with `nvidia-smi topo -m` (showing NODE/PIX paths
   reachable for P2P) or the engine log's P2P banner.

3. Image. The serving image is Jake Tempo's
   [jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek)
   source tree (rev `bb386d39098e…`) rebuilt on amd64 for SM120. One
   wrapper clones, verifies, and applies the overlay:

   ```bash
   bash reproduce/prepare-tempo-tree.sh /path/to/tempo-tree
   cd /path/to/tempo-tree
   bash build.sh   # re-verifies revision, clean tree, base digest; then builds
   ```

   `reproduce/tempo-overlay/` holds the four files the wrapper copies over
   upstream: `stage.py` changes `ARCH_LIST` from `12.1a` to `12.0a` (RTX
   PRO 6000 is SM120; the GB10 Spark is not) and raises `MAX_JOBS` 1 to 16
   with 4 NVCC threads. The Dockerfile pins the base by digest
   (`vllm/vllm-openai@sha256:00d577a6…`, what the mutable
   `deepseekv41-flash-0909` tag pointed at when this was built) and
   installs the x86_64 cmake wheel; `sources-amd64.json` carries the same
   archive pins as upstream with the amd64 cmake wheel. The build runs
   five source stages; at MAX_JOBS 16 it took about an hour on a 60-core
   host (upstream's MAX_JOBS=1 defaults are far slower).

   A prebuilt image from these exact files is on GHCR (built 2026-09-21,
   public; anonymous pull verified against the registry digest):

   ```bash
   docker pull ghcr.io/mark-yong/dsv41-tempo-sm120-tp3@sha256:075c9cd7d4194e931a10a2be7dbdd7ad499f736e07869ea66b39b85963aea448
   # or: docker pull ghcr.io/mark-yong/dsv41-tempo-sm120-tp3:amd64
   ```

   The source build above remains the reference; the pull is a courtesy
   artifact.

Community image status (filled per the Local Inference Lab Community
Docker Publishing Checklist; this image is documented here only and is
not announced or supported in the community):

- Status: experimental community derivative; not maintained
- Image and digest: `ghcr.io/mark-yong/dsv41-tempo-sm120-tp3@sha256:075c9cd7d4194e931a10a2be7dbdd7ad499f736e07869ea66b39b85963aea448`
- Based on: [jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek) @ `bb386d39098e…` on `vllm/vllm-openai@sha256:00d577a6a632…`
- Build recipe: this repo, `reproduce/` (`prepare-tempo-tree.sh`, then `build.sh`)
- Source commits and patches: `reproduce/tempo-overlay/sources-amd64.json`
  pins all seven source archives (vLLM `e47aa780…`, cuda-exl3 `6a1ffc34…`,
  FlashInfer `07869c61…`, CUTLASS ×2, CCCL, spdlog) and the 15-overlay
  Tempo patch set; the amd64 delta is exactly the four files in
  `reproduce/tempo-overlay/`
- Changes from base: amd64/SM120 port (`ARCH_LIST` 12.1a→12.0a), parallel
  build (MAX_JOBS 16), x86_64 cmake wheel; no engine-behavior changes
  beyond Tempo's own patch set
- B12X: N/A — the EXL3 path does not use the B12X kernel backend
- Tested configuration: 3× RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16,
  NODE topology; NVIDIA driver 615.71.09, CUDA 13.4 user mode; TP3;
  Pollard 3.51 bpw EXL3 @ `f129e31a…`; fp8 KV (4 GiB); PIECEWISE CUDA
  graphs; DSpark speculative (`num_speculative_tokens=5`); 32,768
  max-model-len; compose defaults in `reproduce/compose/`
- Validation results: `llm-inference-bench` decode matrix (conc 1/2/4 ×
  context 0/16k, 30 s sustained, 2048 max output tokens) and the prefill
  ladder at 8k/16k, during an exclusive GPU window; commands in
  Reproducing step 5
- Known limitations: no long-context prefill (131k boots but OOMs under
  bench); prefill and decode trail the dense+UVA reference (external
  numbers, not an A/B); DSpark net speedup not isolated; 8 GiB KV not
  usable (see "What failed"); fidelity checks were smoke tests and the
  bench tables only
- Support: none committed. Issues on this repository are accepted but may
  go unanswered; the author runs this path as a documented dead end (see
  Takeaways).

4. Serve. The compose file reproduces the keep config directly:

   ```bash
   cd <this repo>/reproduce/compose
   cp .env.example .env    # set VLLM_API_KEY_DSV41 and MODEL_HOST_PATH
   docker compose up -d    # stop other GPU tenants first; exclusive window
   curl -s -H "Authorization: Bearer $VLLM_API_KEY_DSV41" \
     http://localhost:8014/v1/models
   ```

   Defaults are the P4 values from the tables above (32k context, 4 GiB KV,
   seqs 4, batched 4096, util 0.75, Engram pinned DDR, custom all-reduce
   on, P2P on). The compose file expects an image tagged
   `dsv41-tempo-sm120-tp3:amd64` (default `IMAGE=`) and does not pull
   anything; if you use the GHCR image instead, pull it (public, digest
   above) and either retag it to that name or set `IMAGE=` in `.env` to
   the digest reference. `docker-compose.yml` header comments map the
   P0-P3 ladder steps to the flags that differ.

5. Bench. Numbers came from
   [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
   (Martin Vit) at commit `d115feee75095081bda2520aa046986a8885f449`
   (2026-09-01): 30 s sustained decode per cell, 2048 max output tokens,
   decode concurrencies 1/2/4 at context 0 and 16k, engine-default
   sampling. `reproduce/tools/summarize-prefill.py` reduces a bench result
   JSON to one line per context;
   `reproduce/tools/capture-vram.sh <dir>` snapshots `nvidia-smi` and free
   RAM per step.

## Related reading

- [peterkilfeather's decoder-half UVA offload gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed):
  the dense-weights path this page compares against; vLLM UVA overlay and
  compose, measured from 32k through 1M
- Jake Tempo / Spark TP3 EXL3 lineage (links in the table above)

---

*Recorded 2026-09-21. Hardware: 3× RTX PRO 6000 96 GB SM120 on PCIe 4.0
x16, NODE topology (no NVLink/P2P at the fabric level; P2P forced in
software for the all-reduce step). Numbers from `llm-inference-bench`
during an exclusive GPU window.*
