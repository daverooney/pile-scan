"""Running Gemma 4 through llama.cpp's own server.

Three jobs, each safe to repeat at the start of every Colab session:

1. `check_environment` and `ensure_server_binary`: confirm the runtime can run
   llama.cpp's prebuilt CUDA build, then download and unpack it (no compiling).
   `verify_server` then confirms it can actually see the GPU.
2. `JitCache`: the prebuilt build has no ready-made T4 code, so the GPU driver
   compiles it on first use. The result is saved to Drive so this happens once.
3. `ensure_model_files`: download the GGUF model and its vision projector.
4. `LlamaServer` and `LlamaPageReader`: start the server, then send it one page
   at a time over its OpenAI-style HTTP API.

Server settings that matter for Gemma 4 (checked against llama.cpp v1.0.0's
source, common/arg.cpp, and https://dev.to/someoddcodeguy/a-quick-note-on-gemma-4-image-settings-in-llamacpp-39ng):

* `--image-min-tokens` / `--image-max-tokens` set the visual token budget.
  Both are set to the configured value (1120 by default) so every page gets
  the full budget.
* `-b` and `-ub` must be at least as large as one image's tokens, because
  Gemma 4 encodes an image in a single non-causal pass. With the default
  ubatch of 512 the server aborts with "non-causal attention requires
  n_ubatch >= n_tokens". We use 2048.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import extract, strips
from .config import LLAMA_CPP_FALLBACK_BUILD, ScanConfig
from .loops import find_loop
from .reader import ReadResult

BATCH_SIZE = 2048
RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
LATEST_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# The prebuilt Linux CUDA builds are compiled on Ubuntu 24.04 and need glibc
# 2.38 or newer, so they run on Colab's 2026.10 runtime but not on the older
# Ubuntu 22.04 runtimes (glibc 2.35).
MIN_GLIBC = (2, 38)
# A CUDA 12.8 build needs an R570 or newer driver. Colab's T4s report 580.
MIN_DRIVER = 570

Runner = Callable[..., subprocess.CompletedProcess]


class EnvironmentProblem(RuntimeError):
    """The Colab runtime can't run the prebuilt server. The message says how to fix it."""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


# --- checking the runtime ------------------------------------------------------

@dataclass
class GpuInfo:
    name: str
    driver: str
    compute_cap: str     # "7.5" for a T4, "8.9" for an L4
    memory_mib: int

    @property
    def needs_jit(self) -> bool:
        """True when the prebuilt build has no native code for this GPU.

        The CUDA 12.8 release ships native code for compute 8.6, 8.9 and 12.0
        and PTX for the rest (llama.cpp's default architecture list in
        ggml/src/ggml-cuda/CMakeLists.txt), so a T4 (7.5) needs the driver to
        compile kernels on first use.
        """
        return self.compute_cap.replace(".", "") not in ("86", "89", "120")


def glibc_version() -> tuple[int, int]:
    try:
        name, ver = os.confstr("CS_GNU_LIBC_VERSION").split()
        major, minor = ver.split(".")[:2]
        return int(major), int(minor)
    except (ValueError, OSError, AttributeError):
        return (0, 0)


def gpu_info(runner: Runner = _run) -> GpuInfo | None:
    try:
        out = runner(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    line = out.strip().splitlines()[0]
    name, driver, cap, mem = (x.strip() for x in line.split(","))
    return GpuInfo(name=name, driver=driver, compute_cap=cap, memory_mib=int(float(mem)))


def check_environment(cfg: ScanConfig, *, runner: Runner = _run, glibc: tuple[int, int] | None = None,
                      log=print) -> GpuInfo:
    """Fail early, with a fix, if this runtime can't run the prebuilt server."""
    glibc = glibc or glibc_version()
    if glibc < MIN_GLIBC:
        raise EnvironmentProblem(
            f"This Colab runtime has glibc {glibc[0]}.{glibc[1]}, but llama.cpp's prebuilt CUDA build "
            f"needs {MIN_GLIBC[0]}.{MIN_GLIBC[1]} or newer (Ubuntu 24.04). Fix: Runtime > Change runtime "
            "type > Runtime version > Latest (2026.10 or newer), then Run all again."
        )
    gpu = gpu_info(runner)
    if gpu is None:
        raise EnvironmentProblem("No GPU found. Runtime > Change runtime type > pick a GPU (T4 or L4), "
                                 "then Run all again.")
    try:
        driver_major = int(gpu.driver.split(".")[0])
    except ValueError:
        driver_major = 0
    if driver_major < MIN_DRIVER:
        raise EnvironmentProblem(
            f"GPU driver {gpu.driver} is too old for the CUDA {cfg.llama_cpp_cuda} build "
            f"(needs {MIN_DRIVER} or newer). Switch to the latest runtime version."
        )
    # Weights in GiB, plus about 3.5 GiB for the vision projector, KV cache
    # and the 2048-token compute buffers.
    weights_gib = cfg.weights_gib
    if weights_gib and weights_gib + 3.5 > gpu.memory_mib / 1024:
        log(f"Warning: {cfg.model} {cfg.quantization} may not fit on this {gpu.name} "
            f"({gpu.memory_mib / 1024:.0f} GiB). If the server runs out of memory, use 4bit or an L4.")
    log(f"Runtime OK: glibc {glibc[0]}.{glibc[1]}, {gpu.name} ({gpu.memory_mib} MiB), "
        f"driver {gpu.driver}, compute {gpu.compute_cap}.")
    return gpu


# --- getting the prebuilt server -------------------------------------------------

def release_assets(build: str, cuda: str) -> list[str]:
    """URLs of the server tarball and the CUDA runtime libraries it links against."""
    return [
        f"{RELEASES}/{build}/llama-{build}-bin-ubuntu-cuda-{cuda}-x64.tar.gz",
        f"{RELEASES}/{build}/cudart-llama-{build}-bin-ubuntu-cuda-{cuda}-x64.tar.gz",
    ]


def _extract_flat(archive: Path, dest: Path) -> None:
    """Unpack a release tarball's files directly into `dest` (dropping its top folder).

    The server finds its libraries next to itself (RUNPATH $ORIGIN), so the
    CUDA runtime tarball is unpacked into the same folder.
    """
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not (member.isfile() or member.islnk() or member.issym()):
                continue
            name = Path(member.name).name
            if not name or name in (".", ".."):
                continue
            target = dest / name
            if member.issym():
                link = Path(member.linkname).name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(link)
                continue
            src = tf.extractfile(member)   # follows hard links too
            if target.exists() or target.is_symlink():
                target.unlink()
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "pile-scan"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _build_number(build: str) -> int:
    m = re.fullmatch(r"b(\d+)", build)
    return int(m.group(1)) if m else -1


def release_cache_dir(cfg: ScanConfig, build: str) -> Path:
    """Where a release's two archives are kept on Drive."""
    return Path(cfg.cache_dir) / "llama.cpp" / f"{build}-cuda{cfg.llama_cpp_cuda}"


def cached_builds(cfg: ScanConfig) -> list[str]:
    """Builds for this CUDA version that are fully cached on Drive, newest first."""
    root = Path(cfg.cache_dir) / "llama.cpp"
    suffix = f"-cuda{cfg.llama_cpp_cuda}"
    found = []
    if root.is_dir():
        for d in root.iterdir():
            if d.name.endswith(suffix) and (d / ".complete").exists():
                found.append(d.name[: -len(suffix)])
    return sorted(found, key=_build_number, reverse=True)


def resolve_build(cfg: ScanConfig, *, log=print, fetch_json=None) -> tuple[str, str]:
    """Which llama.cpp release to use, and why ("pinned", "cached", "newest", "fallback")."""
    if cfg.llama_cpp_build != "latest":
        return cfg.llama_cpp_build, "pinned"
    cached = cached_builds(cfg)
    if cached:
        return cached[0], "cached"
    fetch_json = fetch_json or _fetch_json
    try:
        release = fetch_json(LATEST_RELEASE_API)
        build = release["tag_name"]
        names = {a.get("name") for a in release.get("assets", [])}
        wanted = {u.rsplit("/", 1)[1] for u in release_assets(build, cfg.llama_cpp_cuda)}
        if names and not wanted <= names:
            log(f"The newest llama.cpp release, {build}, has no CUDA {cfg.llama_cpp_cuda} build yet; "
                f"using {LLAMA_CPP_FALLBACK_BUILD}.")
            return LLAMA_CPP_FALLBACK_BUILD, "fallback"
        return build, "newest"
    except Exception as exc:   # no network to GitHub's API, rate limit, odd reply
        log(f"Couldn't look up the newest llama.cpp release ({exc}); using {LLAMA_CPP_FALLBACK_BUILD}.")
        return LLAMA_CPP_FALLBACK_BUILD, "fallback"


def ensure_server_binary(cfg: ScanConfig, *, log=print, download=None, fetch_json=None) -> Path:
    """Get llama.cpp's prebuilt CUDA server ready on this VM. Returns the llama-server path.

    The release archives are kept on Drive (see `release_cache_dir`), so a new
    session copies them from there instead of downloading. With the build set
    to "latest", the cached release is used as is; delete
    cache_dir/llama.cpp on Drive to move to the newest one.

    Sets `cfg.llama_cpp_build` to the build actually used, so the GPU code
    cache (JitCache) is named after it.
    """
    download = download or _download
    build, source = resolve_build(cfg, log=log, fetch_json=fetch_json)
    cfg.llama_cpp_build = build
    dest = Path(cfg.local_dir) / f"llama-{build}-cuda-{cfg.llama_cpp_cuda}"
    server = dest / "llama-server"
    marker = dest / ".complete"
    drive = release_cache_dir(cfg, build)
    if marker.exists() and server.exists():
        log(f"llama.cpp {build} (CUDA {cfg.llama_cpp_cuda}) already unpacked on this VM.")
        return server
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    t0 = time.time()
    archives = []
    if (drive / ".complete").exists():
        log(f"Using llama.cpp {build} (CUDA {cfg.llama_cpp_cuda}) cached on Drive"
            + (" (delete the cached folder on Drive to move to the newest release)" if source == "cached" else ""))
        archives = [drive / u.rsplit("/", 1)[1] for u in release_assets(build, cfg.llama_cpp_cuda)]
    else:
        what = {"newest": "the newest release", "pinned": "the pinned release",
                "fallback": "the fallback release"}.get(source, "")
        log(f"Downloading llama.cpp {build} (CUDA {cfg.llama_cpp_cuda}), {what}, and caching it on Drive")
        downloads = Path(cfg.local_dir) / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        for url in release_assets(build, cfg.llama_cpp_cuda):
            archive = downloads / url.rsplit("/", 1)[1]
            if not archive.exists():
                download(url, archive, log=log)
            archives.append(archive)
    for archive in archives:
        _extract_flat(archive, dest)
    if not server.exists():
        raise RuntimeError(f"{server} missing after unpacking the llama.cpp release.")
    if not (drive / ".complete").exists():
        try:
            drive.mkdir(parents=True, exist_ok=True)
            for archive in archives:
                tmp = drive / (archive.name + ".partial")
                shutil.copyfile(archive, tmp)
                os.replace(tmp, drive / archive.name)
            (drive / ".complete").write_text(build)
        except OSError as exc:
            log(f"Couldn't cache llama.cpp on Drive ({exc}); the next session will download it again.")
        for archive in archives:
            archive.unlink(missing_ok=True)
    marker.write_text(build)
    log(f"llama.cpp {build} (CUDA {cfg.llama_cpp_cuda}) ready in {time.time() - t0:.0f}s")
    return server


def verify_server(binary: Path, *, env: dict | None = None, runner: Runner = _run) -> str:
    """Run `llama-server --list-devices` and make sure it sees a CUDA GPU.

    The prebuilt build loads its CUDA backend at run time; if that fails (a
    missing library, a driver mismatch) it quietly falls back to the CPU, which
    would be unusably slow. This turns that into a clear error.
    """
    try:
        res = runner([str(binary), "--list-devices"], capture_output=True, text=True,
                     env=env, timeout=600, check=False)
    except OSError as exc:
        raise EnvironmentProblem(f"Could not run {binary}: {exc}") from None
    text = (res.stdout or "") + (res.stderr or "")
    if "GLIBC_" in text and "not found" in text:
        raise EnvironmentProblem(
            "llama-server needs a newer glibc than this runtime has. Runtime > Change runtime type > "
            "Runtime version > Latest, then Run all again.\n" + text.strip()[-500:])
    cuda_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("CUDA")]
    if res.returncode != 0 or not cuda_lines:
        hint = _missing_libraries(binary.parent / "libggml-cuda.so", runner)
        raise EnvironmentProblem(
            "llama-server can't see the GPU, so it would run on the CPU." + hint +
            "\nOutput of --list-devices:\n" + text.strip()[-1000:])
    return cuda_lines[0]


def _missing_libraries(lib: Path, runner: Runner) -> str:
    try:
        out = runner(["ldd", str(lib)], capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""
    missing = [ln.split("=>")[0].strip() for ln in out.splitlines() if "not found" in ln]
    return f" Missing libraries: {', '.join(missing)}." if missing else ""


# --- the T4's first-launch GPU code cache -------------------------------------------

class JitCache:
    """Keeps the CUDA driver's compiled-kernel cache on Drive between sessions.

    On a GPU the prebuilt build has no native code for (the T4), the driver
    compiles each kernel from PTX the first time it's used and stores the
    result in CUDA_CACHE_PATH. That folder lives on local disk while the server
    runs (reading it over the Drive mount would be slow) and is saved to Drive
    as one tar file. The cache is specific to the GPU model, driver and
    llama.cpp build, so all three are in the file name.
    """

    def __init__(self, cfg: ScanConfig, gpu: GpuInfo, *, log=print):
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", f"{gpu.name}-{gpu.driver}").strip("-")
        self.local = Path(cfg.local_dir) / "cuda-jit-cache"
        self.drive_file = (Path(cfg.cache_dir) / "cuda-jit" /
                           f"{cfg.llama_cpp_build}-cuda{cfg.llama_cpp_cuda}-{slug}.tar")
        self.log = log
        self._saved_signature: tuple[int, int] | None = None

    def env(self) -> dict[str, str]:
        """Environment variables for the server process."""
        return {
            **os.environ,
            "CUDA_CACHE_PATH": str(self.local),
            "CUDA_CACHE_MAXSIZE": str(4 * 1024 ** 3),   # the maximum; the default 256 MiB is too small
            "CUDA_CACHE_DISABLE": "0",
        }

    def _signature(self) -> tuple[int, int]:
        files = [p for p in self.local.rglob("*") if p.is_file()] if self.local.exists() else []
        return len(files), sum(p.stat().st_size for p in files)

    def restore(self) -> bool:
        self.local.mkdir(parents=True, exist_ok=True)
        if not self.drive_file.exists():
            return False
        try:
            with tarfile.open(self.drive_file) as tf:
                for member in tf.getmembers():
                    rel = Path(member.name)
                    if not member.isfile() or rel.is_absolute() or ".." in rel.parts:
                        continue
                    target = self.local / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
        except (tarfile.TarError, OSError) as exc:
            self.log(f"Couldn't read the saved GPU code ({exc}); it will be compiled again.")
            return False
        self._saved_signature = self._signature()
        self.log(f"Restored compiled GPU code from {self.drive_file.name} "
                 f"({self._saved_signature[1] / 1e6:.0f} MB)")
        return True

    def save(self) -> bool:
        """Copy the cache to Drive if it changed since the last save. Returns True if saved.

        Cheap when nothing changed, so it's called after every page: on a T4
        the first few pages compile new kernels, then the cache stops growing.
        Never raises; a failed save only means compiling again next session.
        """
        try:
            return self._save()
        except OSError as exc:
            self.log(f"Couldn't save compiled GPU code to Drive: {exc}")
            return False

    def _save(self) -> bool:
        sig = self._signature()
        if sig[0] == 0 or sig == self._saved_signature:
            return False
        self.drive_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.drive_file.with_name(f"{self.drive_file.name}.tmp-{os.getpid()}")
        with tarfile.open(tmp, "w") as tf:
            for p in sorted(self.local.rglob("*")):
                if p.is_file():
                    tf.add(p, arcname=str(p.relative_to(self.local)))
        os.replace(tmp, self.drive_file)
        self._saved_signature = sig
        self.log(f"Saved compiled GPU code to Drive ({sig[1] / 1e6:.0f} MB)")
        return True


# --- model files -------------------------------------------------------------

def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _download(url: str, dest: Path, *, token: str | None = None, log=print) -> None:
    """Stream `url` to `dest`, resuming a partial download if one is there."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "pile-scan"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:   # server ignored Range: start over
            have = 0
        total = have + int(resp.headers.get("Content-Length") or 0)
        mode = "ab" if have else "wb"
        last = time.time()
        with open(part, mode) as fh:
            done = have
            while chunk := resp.read(8 << 20):
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last > 15:
                    log(f"  {dest.name}: {done / 1e9:.1f} / {total / 1e9:.1f} GB")
                    last = time.time()
    os.replace(part, dest)


@dataclass
class ModelFiles:
    model: Path
    mmproj: Path
    mtp: Path | None = None   # the MTP drafter, when cfg.use_mtp


def ensure_model_files(cfg: ScanConfig, *, token: str | None = None, log=print,
                       download=_download) -> ModelFiles:
    """Make sure the model, vision projector and (if used) MTP drafter are on local disk."""
    # One folder per repo: the 12B and E4B both call their projector mmproj-F16.gguf.
    repo_slug = cfg.repo.replace("/", "__")
    model_dir = Path(cfg.model_dir) / repo_slug
    drive_dir = Path(cfg.cache_dir) / "models" / repo_slug
    names = [cfg.model_file, cfg.mmproj]
    if cfg.use_mtp:
        names.append(cfg.mtp_file)
    out = []
    for name in names:
        local = model_dir / name
        cached = drive_dir / name
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size > 0:
            pass
        elif cfg.cache_models_on_drive and cached.exists():
            log(f"Copying {name} from Drive cache")
            shutil.copyfile(cached, local)
        else:
            log(f"Downloading {name} from {cfg.repo}")
            download(hf_url(cfg.repo, name), local, token=token, log=log)
            if cfg.cache_models_on_drive:
                cached.parent.mkdir(parents=True, exist_ok=True)
                tmp = cached.with_name(f"{cached.name}.tmp-{os.getpid()}")
                shutil.copyfile(local, tmp)
                os.replace(tmp, cached)
        out.append(local)
    return ModelFiles(model=out[0], mmproj=out[1], mtp=out[2] if cfg.use_mtp else None)


# --- the server --------------------------------------------------------------

def server_args(binary: Path, files: ModelFiles, cfg: ScanConfig) -> list[str]:
    args = [
        str(binary),
        "-m", str(files.model),
        "--mmproj", str(files.mmproj),
        "-ngl", "999",                                   # every layer on the GPU
        "-c", str(cfg.ctx_size),
        "-b", str(BATCH_SIZE), "-ub", str(BATCH_SIZE),   # see module docstring
        "--image-min-tokens", str(cfg.max_soft_tokens),
        "--image-max-tokens", str(cfg.max_soft_tokens),
        "--reasoning-budget", "0",                       # no thinking: we want the transcription
        "-np", "1",                                      # one page at a time
        # The prompt cache keeps each finished request's state in system RAM
        # (8 GiB by default) for reuse. Every page is different, so it never
        # helps, and it filled most of a 12.7 GB Colab VM in 11 pages.
        "--cache-ram", "0",
        "--no-webui",
        "--host", "127.0.0.1", "--port", str(cfg.server_port),
    ]
    if files.mtp is not None:
        args += [
            "--spec-draft-model", str(files.mtp),
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(cfg.mtp_draft_max),
            "--spec-draft-ngl", "all",
        ]
    return args


class LlamaServer:
    """Starts llama-server as a child process and waits until it's ready."""

    def __init__(self, args: list[str], *, port: int, log_path: Path, log=print,
                 env: dict[str, str] | None = None):
        self.args = args
        self.env = env   # e.g. JitCache.env(), so compiled GPU code lands in the cache
        self.port = port
        self.log_path = log_path
        self.log = log
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, timeout: float = 1800) -> "LlamaServer":
        self.stop()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.args, stdout=self._log_fh, stderr=subprocess.STDOUT,
                                     env=self.env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = self.tail()
                bad = re.search(r"error: invalid argument: (\S+)", tail)
                if bad:
                    raise RuntimeError(
                        f"This llama.cpp build doesn't know the option {bad.group(1)}. It may be too old "
                        "for that setting (USE_MTP needs a build from June 2026 or later). Delete the "
                        "llama.cpp folder in the cache folder on Drive to get the newest release, "
                        "or turn the setting off.\n" + tail)
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}. "
                                   f"Last lines of {self.log_path}:\n{tail}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        self.log(f"llama-server ready in {time.time() - t0:.0f}s")
                        return self
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                pass  # 503 while loading, or not listening yet
            time.sleep(2)
        raise TimeoutError(f"llama-server not ready after {timeout:.0f}s. Log:\n{self.tail()}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        fh = getattr(self, "_log_fh", None)
        if fh is not None and not fh.closed:
            fh.close()

    def tail(self, n: int = 30) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def _png_data_uri(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")   # lossless: JPEG artifacts hurt small print
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Used only when re-reading a page that looped. DRY ("don't repeat yourself")
# penalises tokens that extend a sequence already seen. llama.cpp's default
# sequence breakers include newline, which would let a "|"-per-line loop
# through, so "<" (only in our tags) is the sole breaker here.
DRY_SETTINGS = {
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 3,
    "dry_penalty_last_n": 1024,
    "dry_sequence_breakers": ["<"],
}


@dataclass
class _Attempt:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int
    loop_at: int | None = None

    @property
    def cut_off(self) -> bool:
        return self.loop_at is not None or self.finish_reason == "length"

    @property
    def kept_text(self) -> str:
        """The reply without the looping part."""
        return self.text if self.loop_at is None else self.text[: self.loop_at].rstrip()


class LlamaPageReader:
    """Sends one page image + prompt to llama-server and returns the raw reply.

    Replies are streamed so a loop can be stopped as soon as it shows (see
    loops.py) instead of running to the token limit. A page that loops is read
    once more with DRY sampling; a page that hits the limit without looping is
    read once more with `retry_max_new_tokens`. If the second reading is still
    cut off, the longer of the two is kept and the page is marked truncated.
    """

    def __init__(self, server: LlamaServer, cfg: ScanConfig, *, request_timeout: float = 1800, log=print,
                 after_read: Callable[[], object] | None = None, retry: bool = True):
        self.server = server
        self.cfg = cfg
        self.timeout = request_timeout
        self.log = log
        self.retry = retry
        # Called after each page, e.g. JitCache.save, so kernels the T4 compiled
        # for that page reach Drive even if the session dies later.
        self.after_read = after_read
        self._text_only_tokens: dict[str, int] = {}

    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.server.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def _post(self, payload: dict) -> dict:
        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"llama-server returned HTTP {e.code}: {body}") from None

    def _stream(self, payload: dict) -> _Attempt:
        """Stream one reply, stopping early if it starts looping."""
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        length = checked = chunks = 0
        finish = None
        usage: dict = {}
        loop_at = None
        try:
            # Leaving this block closes the connection, which makes llama-server
            # stop generating.
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    obj = json.loads(data)
                    if "error" in obj:
                        raise RuntimeError(f"llama-server error: {json.dumps(obj['error'])[:1000]}")
                    usage = obj.get("usage") or usage
                    for choice in obj.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            parts.append(piece)
                            length += len(piece)
                            chunks += 1
                        finish = choice.get("finish_reason") or finish
                    if length - checked >= 200:
                        checked = length
                        loop_at = find_loop("".join(parts))
                        if loop_at is not None:
                            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"llama-server returned HTTP {e.code}: {body}") from None
        text = "".join(parts)
        if loop_at is None and finish == "length":
            loop_at = find_loop(text)
        return _Attempt(
            text=text,
            finish_reason=finish,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=int(usage.get("completion_tokens") or chunks),
            loop_at=loop_at,
        )

    def _payload(self, content: list[dict], max_tokens: int, **extra) -> dict:
        return {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,     # greedy: transcription wants the likeliest text, not variety
            "chat_template_kwargs": {"enable_thinking": False},
            **extra,
        }

    def _baseline_tokens(self, prompt: str) -> int | None:
        """Prompt tokens for the text alone, so image tokens = total - this."""
        if prompt not in self._text_only_tokens:
            try:
                r = self._post(self._payload([{"type": "text", "text": prompt}], 1))
                self._text_only_tokens[prompt] = int(r["usage"]["prompt_tokens"])
            except Exception:
                return None
        return self._text_only_tokens[prompt]

    def _read_with_retry(self, content: list[dict], max_tokens: int, retry_max_tokens: int
                         ) -> tuple[_Attempt, list[_Attempt], str | None]:
        """One reading, plus one more if it looped or ran out of tokens.

        Returns the attempt to keep, every attempt made, and the retry mode.
        """
        first = self._stream(self._payload(content, max_tokens))
        attempts = [first]
        retry_mode = None
        if first.cut_off and self.retry:
            if first.loop_at is not None:
                retry_mode = "dry"
                second = self._stream(self._payload(content, max_tokens, **DRY_SETTINGS))
            else:
                retry_mode = "longer"
                second = self._stream(self._payload(content, retry_max_tokens))
            attempts.append(second)
        final = attempts[-1]
        if final.cut_off and len(attempts) > 1 and len(first.kept_text) > len(final.kept_text):
            final = first
        return final, attempts, retry_mode

    def _image_tokens(self, attempts: list[_Attempt], prompt: str, n_images: int = 1) -> int | None:
        prompt_tokens = next((a.prompt_tokens for a in attempts if a.prompt_tokens is not None), None)
        base = self._baseline_tokens(prompt)
        if prompt_tokens is None or base is None:
            return None
        return (prompt_tokens - base) // n_images

    def read(self, image, prompt: str) -> ReadResult:
        t0 = time.time()
        if self.cfg.strips > 1:
            result = self._read_in_strips(image)
        else:
            content = [
                {"type": "image_url", "image_url": {"url": _png_data_uri(image)}},  # image first, per Google
                {"type": "text", "text": prompt},
            ]
            final, attempts, retry_mode = self._read_with_retry(
                content, self.cfg.max_new_tokens, self.cfg.retry_max_new_tokens)
            result = ReadResult(
                raw=final.kept_text,
                output_tokens=sum(a.completion_tokens for a in attempts),
                image_soft_tokens=self._image_tokens(attempts, prompt),
                elapsed_s=0.0,
                truncated=final.cut_off,
                loop_detected=any(a.loop_at is not None for a in attempts),
                retry_mode=retry_mode,
            )
        result.elapsed_s = time.time() - t0
        if self.after_read is not None:
            self.after_read()
        return result

    def _read_in_strips(self, image) -> ReadResult:
        """Metadata from the whole page, text from horizontal strips (see strips.py).

        Each strip is sent with the whole page first, so the model can see
        column headings that are above the strip; strips inside a ruled table
        also get the table's heading band pasted on top. The pieces are put
        back together as one reply in the usual <meta> + <text> format, and
        each strip's cut and reply are kept in `strips_json`.
        """
        whole = {"type": "image_url", "image_url": {"url": _png_data_uri(image)}}
        meta_prompt = extract.build_meta_prompt(self.cfg.questions)
        meta_final, all_attempts, meta_retry = self._read_with_retry(
            [whole, {"type": "text", "text": meta_prompt}], 1024, 2048)
        image_tokens = self._image_tokens(all_attempts, meta_prompt)
        pieces = strips.make_strips(image, self.cfg.strips)
        texts: list[str] = []
        records: list[dict] = []
        truncated = meta_final.cut_off
        modes = {meta_retry} - {None}
        for piece in pieces:
            prompt = extract.build_strip_prompt(piece.part, len(pieces), heading=piece.heading is not None)
            content = [whole, {"type": "image_url", "image_url": {"url": _png_data_uri(piece.image)}},
                       {"type": "text", "text": prompt}]
            final, attempts, mode = self._read_with_retry(
                content, self.cfg.max_new_tokens, self.cfg.retry_max_new_tokens)
            all_attempts += attempts
            if mode:
                modes.add(mode)
            text, closed = extract.text_section(final.kept_text)
            texts.append(text)
            truncated = truncated or final.cut_off or not closed
            records.append({**piece.record(), "reply": final.kept_text,
                            "truncated": final.cut_off or not closed,
                            "loop": any(a.loop_at is not None for a in attempts), "retry_mode": mode,
                            "output_tokens": sum(a.completion_tokens for a in attempts)})
        joined, dropped = extract.join_strips(texts)
        for rec, gone in zip(records, dropped):
            rec["dropped_lines"] = gone
        meta = extract.meta_section(meta_final.kept_text)
        raw = f"{meta}\n<text>\n{joined}" + ("" if truncated else "\n</text>")
        return ReadResult(
            raw=raw,
            output_tokens=sum(a.completion_tokens for a in all_attempts),
            image_soft_tokens=image_tokens,
            elapsed_s=0.0,
            truncated=truncated,
            loop_detected=any(a.loop_at is not None for a in all_attempts),
            retry_mode="dry" if "dry" in modes else ("longer" if modes else None),
            strips_json=json.dumps({"meta_reply": meta_final.kept_text, "strips": records},
                                   ensure_ascii=False),
        )

    def recover(self) -> None:
        """Restart the server if it died (e.g. out of GPU memory on a huge page)."""
        if not self.server.alive():
            self.log("llama-server stopped; restarting it.\n" + self.server.tail(10))
            self.server.start()
