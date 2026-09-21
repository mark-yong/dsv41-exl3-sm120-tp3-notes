# Third-party notices carried from Tempo

This build recipe is derived from jakejharris/jspark3-deepseek (Tempo)
revision bb386d39098e582fa1446cb96260cdef1df794f9. The following is
Tempo's THIRD_PARTY_NOTICES.md from that revision, preserved unmodified
per Apache-2.0 section 4(d). Paths inside it refer to the Tempo tree (and
inside the built image), not this repository.

---

# Credits and license boundaries

Version: v2.0.0.

Tempo builds on **tonyd2wild and Kai's DeepSeek Spark serving recipe**, and **bot-lab-21's EXL3 expert checkpoint using WestWaters' Pollard method**. It also depends on DeepSeek-AI's model and bundled DSpark draft, vLLM, turboderp's EXL3 format, cuda-exl3 contributors, FlashInfer and NVIDIA's kernel/toolchain work.

Original Tempo scripts, documentation and modifications are Apache-2.0, copyright 2026 Jake Harris. [LICENSE](LICENSE) covers those contributions. Existing notices and copyrights in copied source remain intact. The files in `patches/` are full modified upstream files, not exclusively Tempo-authored code. They combine Apache-2.0 vLLM source and MIT recipe/plugin changes as applicable.

| Input | Exact source | Applicable notice |
|---|---|---|
| Target model and DSpark draft | DeepSeek-V4.1-Flash `2bc89ac599031fa673cab993f1df02fc4a98c673` | [DeepSeek MIT](licenses/DeepSeek-MIT.txt) |
| EXL3 routed experts | bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard `b60193e0609147553145d1538d935925f2763c1d` | Pinned card metadata MIT; [upstream notices](licenses/EXL3-UPSTREAM-NOTICES.md) |
| Spark recipe / Kai Engram and TP3 patches | tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark `c2c1bb7ee7d2f5faee76c07fd9fc9f27e3a20bc2` | [Tony MIT](licenses/Tony-MIT.txt) |
| vLLM | vllm-project/vllm `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` | [Apache-2.0](licenses/vllm.txt) |
| cuda-exl3 | Zeuss5/cuda-exl3 `6a1ffc34866e23f484574ce1922a8bca93eb33b2` | [MIT](licenses/cuda-exl3.txt) |
| FlashInfer, CUTLASS, CCCL, spdlog | Exact commits in [sources.json](release/sources.json) | Individual texts under [licenses](licenses/) |

The upstream EXL3 card says shards 1,2,43–48 are byte-identical to DeepSeek's release; the bundled MXFP4 DSpark drafter is in these unchanged components. Tempo's model file and range checks verify the bytes actually used. It does not create new trained weights or a new quantization. The expert quantization itself is bot-lab-21's work, using EXL3/Pollard methods.

The original model and draft are downloaded together from the exact EXL3 snapshot. There is no Inco DFlash dependency or non-commercial DFlash license carried over from Cadence. This repository does not distribute weight files. A user downloads them directly from upstream and remains responsible for applicable upstream terms.

The base vLLM/NVIDIA image includes additional third-party components (CUDA, NCCL, PyTorch and other libraries) under their own bundled notices. Recipe licensing does not relicense those components or make a legal-use certification. The upstream EXL3 notice file describes its broader upstream author's experiments; components such as b12x are not automatically part of Tempo merely because that notice mentions them. Tempo's actual runtime is fixed by its manifest and source/build inputs.
