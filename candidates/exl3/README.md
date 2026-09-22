# DeepSeek-V4.1-Flash EXL3 TP3 on 3× RTX PRO 6000 (SM120)

This repo records a bring-up of the Pollard 3.51 bpw EXL3 checkpoint of
DeepSeek-V4.1-Flash on 3× RTX PRO 6000 96 GB (Blackwell SM120) with tensor
parallelism 3, plus the performance climb that followed. It covers the config
that measured best, the prefill and decode numbers behind it, and the two
attempts that failed (FlashInfer autotune, 131k context). Successive configs
are labeled P0 through P6. The `candidates/exl3/reproduce/` directory has the image build,
compose file, and helper scripts to run it; runbooks, ops logs, and secrets
stay in my private homelab docs. A later same-box A/B of the P4 keep
config against dense decoder-half UVA is in [COMPARE.md](../uva/receipts/COMPARE.md).

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
(layers 20 and up) in pinned host RAM, DSpark off. Gist provenance: PCIe
Gen5 x16/x16/x8 host, 377 GiB RAM, dense `fb2764a5` checkpoint, LIL r38
serving image; this box is PCIe 4.0 x16 NODE, so the gist's absolute
numbers are not expected to transfer 1:1 (prefill reads the offloaded
experts over PCIe). The P4 peak here is
~6.1k at 8k and ~5.9k at 16k, and no long-context prefill completed. P4's
gain is measured against this stack's own P0 baseline. Same-box dense UVA
on this Gen4 box (2026-09-22) prefills ~4.3–4.5k at 16k–128k — below P4
at 8k/16k and below Pete's Gen5 6.8–7.4k. Tables: [COMPARE.md](../uva/receipts/COMPARE.md).

### Decode (P4, DSpark on, `llm-inference-bench` sustained)

peterkilfeather's gist measures decode at 76.3 tok/s per user at 32k up to
85.5 at 1M with speculative decoding off. Same-box UVA C=1 decode on this
Gen4 box was 74–76 at 0–32k and 67–72 out to 1M (DSpark off);
[COMPARE.md](../uva/receipts/COMPARE.md). The table below is still EXL3 P4 aggregate
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
3. P4 recovers EXL3 against its own earlier baseline. Same-box, P4 still
   wins short-context prefill (~6.1k @ 8k vs UVA ~4.5k @ 16k). UVA wins
   decode (~75 vs ~55 C=1) and is the path that actually serves 128k / 1M.
   Pete's Gen5 prefill (7,424 @ 32k) does not transfer 1:1 onto this Gen4
   NODE box. Tables: [COMPARE.md](../uva/receipts/COMPARE.md).
4. For long context on this hardware class, dense weights with ordinal UVA
   expert offload (decoder-half experts parked, CED boundary at layer 20)
   is the working path; EXL3 131k OOMs under bench. Same-box UVA decode
   held 67–72 tok/s out to 1M.
5. Engram holds native table weights rather than EXL3 experts; reuse across
   serve images only works for the same Flash revision, and different HF
   cuts need a config match check.
6. On Docker's containerd snapshotter, `docker save` and a naive
   `ctr images export` can produce empty or broken archives. Plan image
   archival with that in mind.

## So what

Who this is useful to: anyone bringing DeepSeek-V4.1-Flash onto a 3×96 GB
Blackwell-class box (RTX PRO 6000, PCIe NODE, no NVLink) and deciding
between the EXL3 path and the dense+UVA expert-offload path.

- If the goal is long context or per-user decode, dense+UVA is the working
  path on this box: C=1 decode 74–76 at 0–32k and 67–72 out to 1M, KV
  pool 2.72M tokens. Short-context prefill still favors EXL3 P4 (~6.1k @
  8k vs ~4.5k UVA). Pete's Gen5 7.4k@32k / 85@1M remains the faster
  prefill/long-decode reference. Same-box tables: [COMPARE.md](../uva/receipts/COMPARE.md).
- If you want EXL3 specifically, the working config and both failure modes
  are documented, so the bring-up cost is the build, not the debugging.
  The measured ceiling here is ~6.1k prefill at 8k and ~55 tok/s per-user
  decode with DSpark on, at 32k max context.
- What EXL3 buys on this box is not demonstrated by this run: the Engram
  tables still spill to pinned host DDR and the KV pool stays small. A
  lower-bpw (~3.25) build to free VRAM was not attempted.

Honest summary: EXL3 TP3 works on SM120 after the Tempo port, and this
records where it lands — a working but not winning configuration. The
reusable parts are the single-knob ladder, the failure anatomy, and the
repro pipeline (manifest-checked checkpoint pin, compose, bench tooling).

## Next steps

Untried, in rough priority order:

1. ~~Same-box A/B against dense+UVA~~ **Done** (2026-09-22):
   [COMPARE.md](../uva/receipts/COMPARE.md). Remaining on that path: standalone
   256k/512k/1M prefill (those cells were decode-only); PYNCCL at TP3 is
   still shared with Pete
   ([b12x#410](https://github.com/local-inference-lab/b12x/issues/410)).
2. DSpark-off decode rung at P4 — isolates the net wall-clock speedup of
   speculation (accept length was 2.3-2.4 tokens/step; the A/B was not run).
3. EXL3 long-context recoveries, one knob at a time: speculative decoding
   off, smaller CUDA graphs, a shorter prefill bench matrix. The 131k bench
   fell short by tens of MiB (474 MiB requested against 417-457 MiB free),
   so any one of these may clear it.
4. A ~3.25 bpw auto-build to free VRAM for a larger KV pool — explicitly
   out of scope so far (see the policy note under Stack); it is the obvious
   lever if EXL3 long context is the goal.
5. A real fidelity check: current validation is smoke tests and the bench
   tables only; nothing measures output quality against dense.
6. Reproduction reports from other topologies — the P2P override step
   assumes a NODE/PCIe layout; an NVLink or GB10 box may differ.

## Not claimed

- Fidelity checks were smoke tests and the bench tables above; nothing else
  was run.
- Nothing here claims 300k or 1M context on EXL3.

## Reproducing

The `candidates/exl3/reproduce/` directory carries the working config and scripts. It
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
   python3 candidates/exl3/reproduce/check-model-manifest.py <dir>   # sizes + SHA-256 vs manifest
   bash candidates/exl3/reproduce/tools/prep-tp3-config.sh           # SRC=<dir>
   ```

   `candidates/exl3/reproduce/model-manifest.json` pins every file of the revision
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
   sudo cp candidates/exl3/reproduce/host/nvidia-p2p-override.conf /etc/modprobe.d/
   sudo update-initramfs -u && sudo reboot
   ```

   Verify after reboot with `nvidia-smi topo -m` (showing NODE/PIX paths
   reachable for P2P) or the engine log's P2P banner.

3. Image. The serving image is Jake Tempo's
   [jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek)
   source tree (rev `bb386d39098e…`) rebuilt on amd64 for SM120. One
   wrapper clones, verifies, and applies the overlay:

   ```bash
   bash candidates/exl3/reproduce/prepare-tempo-tree.sh /path/to/tempo-tree
   cd /path/to/tempo-tree
   bash build.sh   # re-verifies revision, clean tree, base digest; then builds
   ```

   `candidates/exl3/reproduce/tempo-overlay/` holds the four files the wrapper copies over
   upstream: `stage.py` changes `ARCH_LIST` from `12.1a` to `12.0a` (RTX
   PRO 6000 is SM120; the GB10 Spark is not) and raises `MAX_JOBS` 1 to 16
   with 4 NVCC threads. The Dockerfile pins the base by digest
   (`vllm/vllm-openai@sha256:00d577a6…`, what the mutable
   `deepseekv41-flash-0909` tag pointed at when this was built) and
   installs the x86_64 cmake wheel; `sources-amd64.json` carries the same
   archive pins as upstream with the amd64 cmake wheel. The build runs
   five source stages; at MAX_JOBS 16 it took about an hour on a 60-core
   host (upstream's MAX_JOBS=1 defaults are far slower).

   A prebuilt image from these exact files is on GHCR (rebuilt
   2026-09-22 so its `jspark3.sources` label matches the current
   `sources-amd64.json`; public, anonymous pull verified against the
   registry digest):

   ```bash
   docker pull ghcr.io/mark-yong/dsv41f-tempo-sm120-tp3@sha256:ddd31bc723e22f9081228727f148a1faa4f7c6773af0a2bbf4d040584a0b2622
   # or: docker pull ghcr.io/mark-yong/dsv41f-tempo-sm120-tp3:amd64
   ```

   The source build above remains the reference; the pull is a courtesy
   artifact.

Community image status (filled per the Local Inference Lab Community
Docker Publishing Checklist; announced via a writeup link in the LIL
Discord, support still none committed):

- Status: experimental community derivative; not maintained
- Image and digest: `ghcr.io/mark-yong/dsv41f-tempo-sm120-tp3@sha256:ddd31bc723e22f9081228727f148a1faa4f7c6773af0a2bbf4d040584a0b2622`
- Based on: [jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek) @ `bb386d39098e…` on `vllm/vllm-openai@sha256:00d577a6a632…`
- Build recipe: this repo, `candidates/exl3/reproduce/` (`prepare-tempo-tree.sh`, then `build.sh`)
- Source commits and patches: `candidates/exl3/reproduce/tempo-overlay/sources-amd64.json`
  pins all seven source archives (vLLM `e47aa780…`, cuda-exl3 `6a1ffc34…`,
  FlashInfer `07869c61…`, CUTLASS ×2, CCCL, spdlog) and the 15-overlay
  Tempo patch set; the amd64 delta is exactly the four files in
  `candidates/exl3/reproduce/tempo-overlay/`
- Changes from base: amd64/SM120 port (`ARCH_LIST` 12.1a→12.0a), parallel
  build (MAX_JOBS 16), x86_64 cmake wheel; no engine-behavior changes
  beyond Tempo's own patch set
- B12X: N/A — the EXL3 path does not use the B12X kernel backend
- Tested configuration: 3× RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16,
  NODE topology; NVIDIA driver 615.71.09, CUDA 13.4 user mode; TP3;
  Pollard 3.51 bpw EXL3 @ `f129e31a…`; fp8 KV (4 GiB); PIECEWISE CUDA
  graphs; DSpark speculative (`num_speculative_tokens=5`); 32,768
  max-model-len; compose defaults in `candidates/exl3/reproduce/compose/`
- Validation results: `llm-inference-bench` decode matrix (conc 1/2/4 ×
  context 0/16k, 30 s sustained, 2048 max output tokens) and the prefill
  ladder at 8k/16k, during an exclusive GPU window; commands in
  Reproducing step 5
- Known limitations: no long-context prefill (131k boots but OOMs under
  bench); same-box UVA decode beats P4 (~75 vs ~55 C=1) and serves 1M,
  but UVA prefill on this Gen4 box does not beat P4 8k/16k (see
  [COMPARE.md](../uva/receipts/COMPARE.md)); DSpark net speedup not isolated; 8 GiB KV
  not usable (see "What failed"); fidelity checks were smoke tests and
  the bench tables only
- Support: none committed. Issues on this repository are accepted but may
  go unanswered; the author runs this path as a documented dead end (see
  Takeaways).

4. Serve. The compose file reproduces the keep config directly:

   ```bash
   cd <this repo>/candidates/exl3/reproduce/compose
   cp .env.example .env    # set VLLM_API_KEY_DSV41 and MODEL_HOST_PATH
   docker compose up -d    # stop other GPU tenants first; exclusive window
   curl -s -H "Authorization: Bearer $VLLM_API_KEY_DSV41" \
     http://localhost:8014/v1/models
   ```

   Defaults are the P4 values from the tables above (32k context, 4 GiB KV,
   seqs 4, batched 4096, util 0.75, Engram pinned DDR, custom all-reduce
   on, P2P on). The compose file expects an image tagged
   `dsv41f-tempo-sm120-tp3:amd64` (default `IMAGE=`) and does not pull
   anything; if you use the GHCR image instead, pull it (public, digest
   above) and either retag it to that name or set `IMAGE=` in `.env` to
   the digest reference. `docker-compose.yml` header comments map the
   P0-P3 ladder steps to the flags that differ.

5. Bench. Numbers came from
   [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
   (Martin Vit) at commit `d115feee75095081bda2520aa046986a8885f449`
   (2026-09-01): 30 s sustained decode per cell, 2048 max output tokens,
   decode concurrencies 1/2/4 at context 0 and 16k, engine-default
   sampling. `candidates/exl3/reproduce/tools/summarize-prefill.py` reduces a bench result
   JSON to one line per context;
   `candidates/exl3/reproduce/tools/capture-vram.sh <dir>` snapshots `nvidia-smi` and free
   RAM per step.

## Related reading

- [COMPARE.md](../uva/receipts/COMPARE.md): same-box A/B of EXL3 P4 vs dense decoder-half
  UVA vs Pete's Gen5 gist numbers
- [peterkilfeather's decoder-half UVA offload gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed):
  the dense-weights recipe; vLLM UVA overlay and compose, measured from
  32k through 1M on Gen5
- Jake Tempo / Spark TP3 EXL3 lineage (links in the table above)

---

*Recorded 2026-09-21; same-box UVA A/B added 2026-09-22. Hardware: 3× RTX
PRO 6000 96 GB SM120 on PCIe 4.0 x16, NODE topology (no NVLink/P2P at the
fabric level; P2P forced in software for the all-reduce step). Numbers
from `llm-inference-bench` during an exclusive GPU window.*
