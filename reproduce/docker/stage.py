#!/usr/bin/env python3
"""Tempo pinned CPU build stages, executed by the Dockerfile.

Fetch the pinned source archives first; the build.sh script drives the
download and sha256 verification from release/sources-amd64.json. The
build enforces a memory cap, MAX_JOBS=16, and no build-stage network
access. This variant targets RTX PRO 6000 (Blackwell SM120,
ARCH_LIST=12.0a); upstream Tempo targets GB10 (12.1a) with MAX_JOBS=1.
The pinned codeload archives use one leading directory (strip=1).

Stages: vLLM stable extension and Python tree; FlashInfer with pinned
submodules; MXFP8 JIT; sparse MLA JIT; cuda-exl3 plugin. Each stage hashes
its inputs and writes logs plus a receipt to /receipts inside the image.
A failure stops the build. Docker retains completed layer checkpoints.
Final Tempo overlays are applied after all five source stages.

No GPU is used during this build. A fresh runtime smoke is required to
validate kernels on the actual hardware.
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 1 << 20  # streaming read granularity; never loads an archive in RAM
# RTX PRO 6000 Blackwell = sm_120. Tempo upstream uses
# 12.1a for GB10; the variant choice must match the image receipt.
ARCH_LIST = "12.0a"
FLASHINFER_VERSION = "0.7.0rc1"
# Raise compile parallelism for this build (60-core host, >>32 GiB RAM).
# Upstream Tempo uses 1 for Spark memory/repro; native hashes differ either way.
BUILD_CAPS = {
    "MAX_JOBS": "16",
    "NVCC_THREADS": "4",
    "FLASHINFER_NVCC_THREADS": "4",
    "CMAKE_BUILD_PARALLEL_LEVEL": "16",
    "PYTHONUNBUFFERED": "1",
}
# Everything we ever set or log about the environment. Receipts and command
# logs record only these keys; the inherited base-image env stays private.
ENV_ALLOWLIST = frozenset(BUILD_CAPS) | {
    "TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST", "BUILD_NVEP",
    "FLASHINFER_BUILD_NO_PIP", "FLASHINFER_DISABLE_VERSION_CHECK",
    "VLLM_HAS_FLASHINFER_CUBIN",
}
# prewarm5.py semantics: these flip nvcc to debug/lineinfo flags and would
# produce a cache that mismatches the launcher environment.
JIT_DEBUG_VARS = ("FLASHINFER_JIT_VERBOSE", "FLASHINFER_JIT_DEBUG",
                  "FLASHINFER_JIT_LINEINFO")
# Tony build_stable_ext.sh candidate list, resolved by actual file existence.
NVRTC_CANDIDATES = (
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/local/cuda/lib64/libnvrtc.so.*",
    "/usr/local/lib/python3.12/dist-packages/nvidia/*/lib/libnvrtc.so*",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
    "/usr/lib/aarch64-linux-gnu/libnvrtc.so*",
)


class Fail(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def env_delta(env):
    """Allowlisted environment delta only (secrets in inherited env stay out
    of every receipt and log)."""
    return {k: v for k, v in sorted((env or {}).items()) if k in ENV_ALLOWLIST}


# ----------------------------------------------------------------- hashing --


def stream_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_archive(pin, input_dir):
    """Streaming size+sha256 verification BEFORE extraction. Fail-closed."""
    path = Path(input_dir) / pin["file"]
    if not path.is_file():
        raise Fail("missing source archive %s (looked in %s)" % (pin["file"], input_dir))
    size = path.stat().st_size
    if "bytes" in pin and pin["bytes"] is not None and size != pin["bytes"]:
        raise Fail("archive %s size %d != pinned %d (changed source artifact)"
                   % (pin["file"], size, pin["bytes"]))
    sha = stream_sha256(path)
    if sha != pin["sha256"]:
        raise Fail("archive %s sha256 %s != pinned %s (changed source artifact)"
                   % (pin["file"], sha, pin["sha256"]))
    return {"file": pin["file"], "bytes": size, "sha256": sha,
            "verified_utc": utc()}


def extract_archive(pin, input_dir, dest):
    """Strip comes from the pin: git archives (vllm) have no leading dir and
    use 0; codeload tarballs carry <repo>-<sha>/ and use 1."""
    strip = pin.get("strip", 1)
    path = Path(input_dir) / pin["file"]
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", str(path), "-C", str(dest),
                    "--strip-components=%d" % strip], check=True)


# ------------------------------------------------------------- monitoring --


class Monitor:
    """Samples host MemAvailable while compile subprocesses run."""

    def __init__(self):
        self.min_avail = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable:"):
                        b = int(line.split()[1]) * 1024
                        self.min_avail = b if self.min_avail is None \
                            else min(self.min_avail, b)
                        break
            except OSError:
                pass
            self._stop.wait(2.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


class Result:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


class LoggedRunner:
    """Runs subprocesses with output STREAMED live to the receipts log.

    capture=True is reserved for tiny query commands (python -c probes) whose
    stdout we must inspect; everything else writes straight to the log file so
    an observing owner sees compiler progress in real time and no large output
    is buffered in memory.
    """

    def __init__(self, receipts_dir):
        self.log = Path(receipts_dir) / "commands.log"

    def run(self, argv, env=None, cwd=None, check=True, logfile=None,
            capture=False):
        target = Path(logfile) if logfile else self.log
        with target.open("a") as lf:
            lf.write("\n=== %s rc=PENDING cwd=%s capture=%s\n=== ARGV %s\n"
                     "=== ENV-DELTA %s\n"
                     % (utc(), cwd or os.getcwd(), capture, json.dumps(argv),
                        json.dumps(env_delta(env), sort_keys=True)))
            lf.flush()
            if capture:
                p = subprocess.run(argv, env=env, cwd=cwd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
                out = p.stdout or ""
                lf.write(out)
                rc = p.returncode
            else:
                p = subprocess.Popen(argv, env=env, cwd=cwd, stdout=lf,
                                     stderr=subprocess.STDOUT)
                out = ""
                rc = p.wait()
            lf.write("\n=== %s rc=%d\n" % (utc(), rc))
            lf.flush()
        if check and rc != 0:
            raise Fail("subprocess rc=%d: %s" % (rc, argv))
        return Result(rc, out)


def find_in_log(logfile, pattern):
    """Line-oriented regex scan of a (possibly large) log; constant memory."""
    rx = re.compile(pattern)
    with open(logfile, "r", errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                return m
    return None


# --------------------------------------------------------------- receipts --


class Receipts:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def refuse_existing(self, stage):
        p = self.root / ("stage%d.json" % stage)
        if p.exists():
            raise Fail("stage receipt %s already exists; attempts are never "
                       "reused — use a new attempt id" % p)

    def write(self, stage, doc):
        p = self.root / ("stage%d.json" % stage)
        if p.exists():  # never overwrite
            raise Fail("refusing to overwrite existing receipt %s" % p)
        doc = dict(doc)
        doc.update(schema=1, stage=stage, utc=utc(),
                   mono=round(time.monotonic(), 6))
        p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def write_failure(self, stage, reason, extra=None):
        p = self.root / ("stage%d.failure.json" % stage)
        doc = {"schema": 1, "stage": stage, "utc": utc(), "reason": reason,
               "extra": extra or {}}
        p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def log_path(self, stage):
        return self.root / ("stage%d.build.log" % stage)


# ------------------------------------------------------------ cgroup facts --


def cgroup_facts():
    facts = {}
    base = Path("/sys/fs/cgroup")
    for name in ("memory.peak", "memory.current", "memory.max",
                 "memory.events"):
        try:
            facts[name] = (base / name).read_text().strip()
        except OSError:
            facts[name] = None
    return facts


def stage_env(stage, extra=None):
    env = dict(os.environ)
    env.update(BUILD_CAPS)
    if stage in (1, 4, 5):
        env["TORCH_CUDA_ARCH_LIST"] = ARCH_LIST
    if stage in (2, 3, 4):
        env["FLASHINFER_CUDA_ARCH_LIST"] = ARCH_LIST
    if stage == 2:
        env["BUILD_NVEP"] = "0"
        env["FLASHINFER_BUILD_NO_PIP"] = "1"
    if stage == 4:
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        env["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
        for k in JIT_DEBUG_VARS:  # clear debug/JIT contamination
            env.pop(k, None)
    if extra:
        env.update(extra)
    return env


def fresh_dir(path):
    """Work trees must be attempt-fresh: absent or empty."""
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise Fail("work path %s exists and is not empty; refusing dirty tree" % p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================== stage 1: vLLM stable extension ============


def transform_cmake_lists(src_dir):
    """Tony's stable-only transformation, as a pure function.

    - keep/restore an untouched preimage at CMakeLists.txt.orig (cp -n)
    - comment ONLY lines matching ^\\s*include\\(cmake/external_projects/
      with the exact '# STABLE-ONLY BUILD (kai): ' marker
    - returns (preimage_sha, postimage_sha, commented_count); count must be
      frozen (>=1) in the receipt
    """
    cl = Path(src_dir) / "CMakeLists.txt"
    orig = Path(src_dir) / "CMakeLists.txt.orig"
    if not cl.is_file():
        raise Fail("CMakeLists.txt missing under %s" % src_dir)
    if not orig.exists():  # cp -n semantics: keep the FIRST preimage
        shutil.copy2(cl, orig)
    pre = orig.read_bytes()
    pre_sha = hashlib.sha256(pre).hexdigest()
    pat = re.compile(rb"^(\s*)include\(cmake/external_projects/", re.MULTILINE)
    post, n = pat.subn(rb"\1# STABLE-ONLY BUILD (kai): include(cmake/external_projects/", pre)
    if n < 1:
        raise Fail("stable-only transformation matched 0 external_projects "
                   "includes (preimage drifted from pin)")
    cl.write_bytes(post)
    return pre_sha, hashlib.sha256(post).hexdigest(), n


def resolve_nvrtc():
    for cand in NVRTC_CANDIDATES:
        for hit in sorted(globmod.glob(cand)):
            if os.path.isfile(hit):
                return hit
    raise Fail("no libnvrtc.so candidate exists: %s" % (NVRTC_CANDIDATES,))


def run_stage1(ctx):
    pins, runner, rcpt = ctx.pins, ctx.runner, ctx.receipts
    logf = str(rcpt.log_path(1))
    verified = [verify_archive(pins["archives"][k], ctx.input) for k in
                ("vllm", "cutlass-stable")]

    src = fresh_dir(ctx.p("src"))
    extract_archive(pins["archives"]["vllm"], ctx.input, src)
    fresh_dir(ctx.p("src/build"))
    fresh_dir(ctx.p("src/build/_deps"))
    extract_archive(pins["archives"]["cutlass-stable"], ctx.input,
                    ctx.p("src/build/_deps/cutlass-src"))
    marker = str(ctx.p("src/build/_deps/cutlass-src/include/cutlass/cutlass.h"))
    if not os.path.isfile(marker):
        raise Fail("CUTLASS MISSING (no %s)" % marker)

    pre_sha, post_sha, n_inc = transform_cmake_lists(src)

    py = sys.executable
    pypath = runner.run([py, "-c",
                         "import sys;print(':'.join(p for p in sys.path if p))"],
                        logfile=logf, capture=True).stdout.strip()
    torch_prefix = runner.run(
        [py, "-c", "import torch;print(torch.utils.cmake_prefix_path)"],
        logfile=logf, capture=True).stdout.strip()
    nvrtc = resolve_nvrtc()

    configure = [
        "cmake", "-S", str(src), "-B", str(ctx.p("src/build")), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DVLLM_TARGET_DEVICE=cuda",
        "-DVLLM_PYTHON_EXECUTABLE=%s" % py,
        "-DVLLM_PYTHON_PATH=%s" % pypath,
        "-DFETCHCONTENT_BASE_DIR=%s" % ctx.p("src/build/_deps"),
        "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=%s" % ctx.p("src/build/_deps/cutlass-src"),
        "-DCMAKE_PREFIX_PATH=%s" % torch_prefix,
        "-DNVCC_THREADS=4",
        "-DCUDA_nvrtc_LIBRARY=%s" % nvrtc,
    ]
    env1 = stage_env(1)
    with Monitor() as mon:
        runner.run(configure, env=env1, logfile=logf)
        runner.run(["cmake", "--build", str(ctx.p("src/build")),
                    "--target", "_C_stable_libtorch", "-j", "16"],
                   env=env1, logfile=logf)
    sos = sorted(ctx.p("src/build").glob("**/_C_stable_libtorch*.so"))
    if len(sos) != 1:
        raise Fail("expected exactly one _C_stable_libtorch*.so, found %d: %s"
                   % (len(sos), [str(s) for s in sos]))
    stable_so = sos[0]
    stable_sha = stream_sha256(stable_so)

    # Overlay Dockerfile.overlay semantics: merge the pinned python tree into
    # the installed vllm package, drop __pycache__, verify import.
    inst = Path(runner.run(
        [py, "-c", "import vllm, pathlib;"
                   "print(pathlib.Path(vllm.__file__).parent)"],
        logfile=logf, capture=True).stdout.strip())
    pre_inventory = {str(p): stream_sha256(p) for p in inst.rglob("*.so")}
    shutil.copytree(src / "vllm", inst, dirs_exist_ok=True)
    for pyc in inst.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    shutil.copy2(stable_so, inst / stable_so.name)
    post_inventory = {str(p): stream_sha256(p) for p in inst.rglob("*.so")}
    vanished = sorted(set(pre_inventory) - set(post_inventory))
    if vanished:  # preserve other base binaries exactly as the recipe does
        raise Fail("overlay removed pre-existing binaries: %s" % vanished)
    version = runner.run([py, "-c", "import vllm;print(vllm.__version__)"],
                         logfile=logf, capture=True).stdout.strip()

    rcpt.write(1, {
        "kind": "vllm-stable",
        "archives": verified,
        "cmake": {"argv": configure, "nvrtc": nvrtc,
                  "torch_cmake_prefix": torch_prefix},
        "cmakelists": {"preimage_sha256": pre_sha,
                       "postimage_sha256": post_sha,
                       "commented_includes": n_inc},
        "stable_so": {"path": str(inst / stable_so.name),
                      "built_path": str(stable_so), "sha256": stable_sha,
                      "bytes": stable_so.stat().st_size},
        "installed_vllm_dir": str(inst),
        "so_inventory_added": sorted(set(post_inventory) - set(pre_inventory)),
        "so_inventory_changed": sorted(
            p for p in post_inventory
            if p in pre_inventory and pre_inventory[p] != post_inventory[p]),
        "vllm_version": version,
        "min_memavailable_bytes": mon.min_avail,
        "cgroup": cgroup_facts(),
        "build_env": env_delta(env1),
    })


# ============================ stage 2: FlashInfer pinned rebuild ===========


STALE_FI = ("flashinfer-python", "flashinfer-jit-cache", "flashinfer-cubin")
FI_MARKERS = ("3rdparty/cutlass/include/cutlass/cutlass.h",
              "3rdparty/cccl/README.md",
              "3rdparty/spdlog/include/spdlog/spdlog.h")


def run_stage2(ctx):
    pins, runner, rcpt = ctx.pins, ctx.runner, ctx.receipts
    logf = str(rcpt.log_path(2))
    verified = [verify_archive(pins["archives"][k], ctx.input)
                for k in ("flashinfer", "cutlass-fi", "cccl", "spdlog")]

    # Remove stale 0.6.18-family packages, then PROVE absence.
    uninstalled = []
    for name in STALE_FI:
        p = runner.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                        name], check=False, logfile=logf)
        uninstalled.append({"name": name, "rc": p.returncode})
    for name in STALE_FI:
        p = runner.run([sys.executable, "-m", "pip", "show", name],
                       check=False, logfile=logf)
        if p.returncode == 0:
            raise Fail("stale package %s still installed after uninstall" % name)

    fi = fresh_dir(ctx.p("opt/fi-src"))
    extract_archive(pins["archives"]["flashinfer"], ctx.input, fi)
    for sub, key in (("cutlass", "cutlass-fi"), ("cccl", "cccl"),
                     ("spdlog", "spdlog")):
        d = fi / "3rdparty" / sub
        d.mkdir(parents=True, exist_ok=True)
        extract_archive(pins["archives"][key], ctx.input, d)
    for marker in FI_MARKERS:
        if not (fi / marker).is_file():
            raise Fail("flashinfer submodule marker missing: %s" % marker)

    env2 = stage_env(2)
    with Monitor() as mon:
        runner.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--no-deps", "--no-build-isolation", "."],
                   env=env2, cwd=str(fi), logfile=logf)
    pip_list = runner.run([sys.executable, "-m", "pip", "list"],
                          logfile=logf, capture=True).stdout

    verify = (
        "import flashinfer\n"
        "print('FI-VERSION', flashinfer.__version__)\n"
        "from flashinfer.mla import supported_sparse_mla_sm120_configs as f\n"
        "c = f()['dsv4']\n"
        "assert c.supports_decode(num_heads=16, topk=1152), 'dsv4 lacks 16/1152'\n"
        "print('FI-DSV4-OK')\n")
    out = runner.run([sys.executable, "-c", verify], env=env2, logfile=logf,
                     capture=True).stdout
    m = re.search(r"FI-VERSION (\S+)", out)
    if not m:
        raise Fail("flashinfer version line missing from verify output")
    if pins.get("flashinfer_version") and \
            m.group(1) != pins["flashinfer_version"]:
        raise Fail("flashinfer %s != pinned %s" % (m.group(1),
                                                   pins["flashinfer_version"]))
    shutil.rmtree(fi / "build", ignore_errors=True)  # Tony slims the layer

    rcpt.write(2, {
        "kind": "flashinfer-pinned",
        "archives": verified,
        "uninstall": uninstalled,
        "stale_absent": list(STALE_FI),
        "flashinfer_version": m.group(1),
        "pip_inventory": [l for l in pip_list.splitlines()
                          if "flashinfer" in l.lower() or "nvidia-nccl" in l.lower()],
        "install_env": env_delta(env2),
        "min_memavailable_bytes": mon.min_avail,
        "cgroup": cgroup_facts(),
    })


# ==================== stage 3: mxfp8_gemm_cutlass_sm120 .build() ===========


def fi_cache_root():
    # Match the pinned flashinfer/jit/env.py base-directory semantics.
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE", os.path.expanduser("~"))
    return Path(base) / ".cache" / "flashinfer"


def inventory_cache():
    """Discover (never assume) the pinned package's cache tree."""
    root = fi_cache_root()
    files = {}
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files[str(p)] = stream_sha256(p)
    return {"root": str(root), "files": files}


def run_stage3(ctx):
    runner, rcpt = ctx.runner, ctx.receipts
    logf = str(rcpt.log_path(3))
    snippet = (
        "import os\n"
        "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as gen\n"
        "m = gen()\n"
        "assert hasattr(m, 'build'), 'generator lacks build()'\n"
        "m.build(verbose=True)  # build only; build_and_load() deliberately unused\n"
        "ic = getattr(m, 'is_compiled', None)\n"
        "assert ic is not None, 'no is_compiled on module'\n"
        "ic = ic() if callable(ic) else ic\n"
        "glp = getattr(m, 'get_library_path', None)\n"
        "assert glp is not None, 'no get_library_path on module'\n"
        "path = glp() if callable(glp) else glp\n"
        "assert ic, 'is_compiled false after build()'\n"
        "assert path and os.path.isfile(path), 'library path missing: %r' % path\n"
        "print('MXFP8-LIB', path)\n")
    before = inventory_cache()
    with Monitor() as mon:
        runner.run([sys.executable, "-c", snippet],
                   env=stage_env(3), logfile=logf)
    after = inventory_cache()
    m = find_in_log(logf, r"MXFP8-LIB (\S+)")
    if not m:
        raise Fail("mxfp8 build produced no library path output")
    lib = m.group(1)
    lib_sha = stream_sha256(lib)
    new_files = sorted(set(after["files"]) - set(before["files"]))

    rcpt.write(3, {
        "kind": "mxfp8-jit",
        "library": {"path": lib, "sha256": lib_sha,
                    "bytes": os.path.getsize(lib)},
        "cache_before_files": len(before["files"]),
        "cache_after_files": len(after["files"]),
        "cache_new_files": new_files,
        "cache_root": after["root"],
        "build_env": env_delta(stage_env(3)),
        "min_memavailable_bytes": mon.min_avail,
        "cgroup": cgroup_facts(),
    })


# ============ stage 4: sparse_mla_sm120 via the pinned generator ===========


SPARSE_SNIPPET = (
    "import hashlib, inspect, os, re\n"
    "import flashinfer.mla._sparse_mla_sm120 as host\n"
    "src = inspect.getsource(host.get_sparse_mla_sm120_module)\n"
    "print('PINNED-GETTER-SOURCE-BEGIN')\n"
    "print(src)\n"
    "print('PINNED-GETTER-SOURCE-END')\n"
    "from flashinfer.jit import mla as mod\n"
    "assert host.gen_sparse_mla_sm120_module is mod.gen_sparse_mla_sm120_module, 'generator binding drift'\n"
    "print('GENERATOR-MODULE', mod.__name__)\n"
    "print('GENERATOR-FILE', inspect.getfile(mod))\n"
    "with open(inspect.getfile(mod), 'rb') as f:\n"
    "    print('GENERATOR-SHA256', hashlib.sha256(f.read()).hexdigest())\n"
    "gen = getattr(mod, 'gen_sparse_mla_sm120_module')\n"
    "m = gen()\n"
    "assert hasattr(m, 'build'), 'generator module lacks build()'\n"
    "m.build()  # build only; no swallowed exceptions, no load\n"
    "print('SPARSE-BUILT')\n")

MXFP8_RECHECK = (
    "import os\n"
    "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as gen\n"
    "m = gen()\n"
    "ic = getattr(m, 'is_compiled', None)\n"
    "ic = (ic() if callable(ic) else ic) if ic is not None else None\n"
    "glp = getattr(m, 'get_library_path', None)\n"
    "path = (glp() if callable(glp) else glp) if glp is not None else None\n"
    "print('MXFP8-RECHECK is_compiled=%s path=%s exists=%s'\n"
    "      % (ic, path, bool(path) and os.path.isfile(path)))\n"
    "assert ic, 'mxfp8 no longer compiled'\n"
    "assert path and os.path.isfile(path), 'mxfp8 library vanished'\n")


def run_stage4(ctx):
    runner, rcpt = ctx.runner, ctx.receipts
    logf = str(rcpt.log_path(4))
    env4 = stage_env(4)
    cleared = list(JIT_DEBUG_VARS)  # forcibly absent from the compile env
    before = inventory_cache()
    with Monitor() as mon:
        runner.run([sys.executable, "-c", SPARSE_SNIPPET],
                   env=env4, logfile=logf)
        runner.run([sys.executable, "-c", MXFP8_RECHECK],
                   env=env4, logfile=logf)
    after = inventory_cache()
    gen_mod = find_in_log(logf, r"^GENERATOR-MODULE (\S+)")
    gen_file = find_in_log(logf, r"^GENERATOR-FILE (\S+)")
    gen_sha = find_in_log(logf, r"^GENERATOR-SHA256 ([0-9a-f]{64})")
    if not (gen_mod and gen_file and gen_sha):
        raise Fail("sparse generator identity lines missing from output")
    if not find_in_log(logf, r"^SPARSE-BUILT"):
        raise Fail("SPARSE-BUILT marker absent")
    recheck = find_in_log(logf, r"^MXFP8-RECHECK (\S+ \S+ \S+)")
    if not recheck or "is_compiled=True" not in recheck.group(0):
        raise Fail("mxfp8 module no longer compiled after sparse build: %r"
                   % (recheck.group(0) if recheck else None))
    new_files = sorted(set(after["files"]) - set(before["files"]))
    sparse_new = [p for p in new_files if "sparse" in p.lower()]
    if not sparse_new:
        raise Fail("no new sparse_mla artifact appeared in the cache tree; "
                   "build() did not produce the pinned module")

    rcpt.write(4, {
        "kind": "sparse-mla-jit",
        "generator": {"module": gen_mod.group(1), "file": gen_file.group(1),
                      "sha256": gen_sha.group(1)},
        "sparse_new_files": [{"path": p, "sha256": after["files"][p]}
                             for p in sparse_new],
        "mxfp8_recheck": recheck.group(1),
        "jit_debug_vars_cleared": cleared,
        "cache_new_files": new_files,
        "cache_after_files": len(after["files"]),
        "build_env": env_delta(env4),
        "min_memavailable_bytes": mon.min_avail,
        "cgroup": cgroup_facts(),
    })


# ============================ stage 5: cuda-exl3 ===========================


def run_stage5(ctx):
    pins, runner, rcpt = ctx.pins, ctx.runner, ctx.receipts
    logf = str(rcpt.log_path(5))
    verified = [verify_archive(pins["archives"]["cuda-exl3"], ctx.input)]

    exl3 = fresh_dir(ctx.p("opt/cuda-exl3"))
    extract_archive(pins["archives"]["cuda-exl3"], ctx.input, exl3)
    env5 = stage_env(5)
    with Monitor() as mon:
        runner.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--no-deps", "--no-build-isolation", "."],
                   env=env5, cwd=str(exl3), logfile=logf)
    verify = (
        "import flashinfer, torch\n"
        "import cuda_exl3, cuda_exl3._C\n"
        "import pathlib\n"
        "print('EXL3-PKG', pathlib.Path(cuda_exl3.__file__).parent)\n"
        "print('TORCH', torch.__version__, 'CUDA', torch.version.cuda)\n"
        "print('FI', flashinfer.__version__)\n")
    out = runner.run([sys.executable, "-c", verify], env=env5, logfile=logf,
                     capture=True).stdout
    pkg = re.search(r"EXL3-PKG (\S+)", out)
    if not pkg:
        raise Fail("cuda_exl3 import receipt line missing")
    pkg_dir = Path(pkg.group(1))
    so_hashes = {str(p): stream_sha256(p) for p in sorted(pkg_dir.rglob("*.so"))}
    if not any(Path(k).stem.startswith("_C") for k in so_hashes):
        raise Fail("no native _C shared object found under %s" % pkg_dir)

    rcpt.write(5, {
        "kind": "cuda-exl3",
        "archives": verified,
        "package_dir": str(pkg_dir),
        "so_sha256": so_hashes,
        "runtime": {"torch": re.search(r"TORCH (\S+) CUDA (\S+)", out).groups(),
                    "flashinfer": re.search(r"FI (\S+)", out).group(1)},
        "install_env": env_delta(env5),
        "min_memavailable_bytes": mon.min_avail,
        "cgroup": cgroup_facts(),
    })


STAGES = {1: ("vllm-stable", run_stage1), 2: ("flashinfer-pinned", run_stage2),
          3: ("mxfp8-jit", run_stage3), 4: ("sparse-mla-jit", run_stage4),
          5: ("cuda-exl3", run_stage5)}


class Ctx:
    def __init__(self, pins_path, input_dir, receipts_dir, workroot="/"):
        self.pins = json.loads(Path(pins_path).read_text())
        self.input = Path(input_dir)
        self.receipts = Receipts(receipts_dir)
        self.runner = LoggedRunner(self.receipts.root)
        self.workroot = Path(workroot)

    def p(self, *parts):
        return self.workroot.joinpath(*parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="stage.py",
        description="One EXL3 CPU-only build stage inside the capped container")
    ap.add_argument("--stage", required=True, type=int, choices=sorted(STAGES))
    ap.add_argument("--pins", required=True,
                    help="pinned source manifest from fetch_sources.py, copied to /input/pins.json")
    ap.add_argument("--input", default="/input")
    ap.add_argument("--receipts", default="/receipts")
    ap.add_argument("--work-root", default="/",
                    help="root for /src, /opt work trees (tests use a tmp root)")
    args = ap.parse_args(argv)

    kind, fn = STAGES[args.stage]
    rcpt = Receipts(args.receipts)
    rcpt.refuse_existing(args.stage)
    t0 = time.monotonic()
    try:
        ctx = Ctx(args.pins, args.input, args.receipts,
                  workroot=args.work_root)
        fn(ctx)
    except Fail as e:
        rcpt.write_failure(args.stage, str(e))
        print("stage%d (%s) FAILED: %s" % (args.stage, kind, e), file=sys.stderr)
        return 3
    except Exception as e:  # unexpected: preserve evidence, fail loudly
        rcpt.write_failure(args.stage, "unexpected: %r" % e)
        print("stage%d (%s) UNEXPECTED FAILURE: %r" % (args.stage, kind, e),
              file=sys.stderr)
        return 3
    print("stage%d (%s) OK elapsed=%.1fs" % (args.stage, kind,
                                             time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
