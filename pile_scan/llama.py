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

from .config import ScanConfig
from .reader import ReadResult

BATCH_SIZE = 2048
RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"

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
    # File sizes in GiB, plus about 3.5 GiB for the vision projector, KV cache
    # and the 2048-token compute buffers.
    weights_gib = {"4bit": 6.9, "8bit": 11.8}.get(cfg.quantization)
    if weights_gib and weights_gib + 3.5 > gpu.memory_mib / 1024:
        log(f"Warning: {cfg.quantization} may not fit on this {gpu.name} "
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


def ensure_server_binary(cfg: ScanConfig, *, log=print, download=None) -> Path:
    """Download and unpack llama.cpp's prebuilt CUDA server. Returns the llama-server path.

    Skips the download if this VM already has it unpacked.
    """
    download = download or _download
    dest = Path(cfg.local_dir) / f"llama-{cfg.llama_cpp_build}-cuda-{cfg.llama_cpp_cuda}"
    server = dest / "llama-server"
    marker = dest / ".complete"
    if marker.exists() and server.exists():
        return server
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    downloads = Path(cfg.local_dir) / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for url in release_assets(cfg.llama_cpp_build, cfg.llama_cpp_cuda):
        archive = downloads / url.rsplit("/", 1)[1]
        if not archive.exists():
            log(f"Downloading {archive.name}")
            download(url, archive, log=log)
        _extract_flat(archive, dest)
        archive.unlink()
    if not server.exists():
        raise RuntimeError(f"{server} missing after unpacking the llama.cpp release.")
    marker.write_text(cfg.llama_cpp_build)
    log(f"llama.cpp {cfg.llama_cpp_build} (CUDA {cfg.llama_cpp_cuda}) ready in {time.time() - t0:.0f}s")
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


def ensure_model_files(cfg: ScanConfig, *, token: str | None = None, log=print,
                       download=_download) -> tuple[Path, Path]:
    """Make sure the model and vision projector are on local disk. Returns their paths."""
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    drive_dir = Path(cfg.cache_dir) / "models" / cfg.gguf_repo.replace("/", "__")
    out = []
    for name in (cfg.model_file, cfg.mmproj_file):
        local = model_dir / name
        cached = drive_dir / name
        if local.exists() and local.stat().st_size > 0:
            pass
        elif cfg.cache_models_on_drive and cached.exists():
            log(f"Copying {name} from Drive cache")
            shutil.copyfile(cached, local)
        else:
            log(f"Downloading {name} from {cfg.gguf_repo}")
            download(hf_url(cfg.gguf_repo, name), local, token=token, log=log)
            if cfg.cache_models_on_drive:
                drive_dir.mkdir(parents=True, exist_ok=True)
                tmp = drive_dir / f"{name}.tmp-{os.getpid()}"
                shutil.copyfile(local, tmp)
                os.replace(tmp, cached)
        out.append(local)
    return out[0], out[1]


# --- the server --------------------------------------------------------------

def server_args(binary: Path, model: Path, mmproj: Path, cfg: ScanConfig) -> list[str]:
    return [
        str(binary),
        "-m", str(model),
        "--mmproj", str(mmproj),
        "-ngl", "999",                                   # every layer on the GPU
        "-c", str(cfg.ctx_size),
        "-b", str(BATCH_SIZE), "-ub", str(BATCH_SIZE),   # see module docstring
        "--image-min-tokens", str(cfg.max_soft_tokens),
        "--image-max-tokens", str(cfg.max_soft_tokens),
        "--reasoning-budget", "0",                       # no thinking: we want the transcription
        "-np", "1",                                      # one page at a time
        "--no-webui",
        "--host", "127.0.0.1", "--port", str(cfg.server_port),
    ]


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
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}. "
                                   f"Last lines of {self.log_path}:\n{self.tail()}")
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


class LlamaPageReader:
    """Sends one page image + prompt to llama-server and returns the raw reply."""

    def __init__(self, server: LlamaServer, cfg: ScanConfig, *, request_timeout: float = 1800, log=print,
                 after_read: Callable[[], object] | None = None):
        self.server = server
        self.cfg = cfg
        self.timeout = request_timeout
        self.log = log
        # Called after each page, e.g. JitCache.save, so kernels the T4 compiled
        # for that page reach Drive even if the session dies later.
        self.after_read = after_read
        self._text_only_tokens: dict[str, int] = {}

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.server.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"llama-server returned HTTP {e.code}: {body}") from None

    def _payload(self, content: list[dict], max_tokens: int) -> dict:
        return {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,     # greedy: transcription wants the likeliest text, not variety
            "chat_template_kwargs": {"enable_thinking": False},
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

    def read(self, image, prompt: str) -> ReadResult:
        t0 = time.time()
        content = [
            {"type": "image_url", "image_url": {"url": _png_data_uri(image)}},  # image first, per Google
            {"type": "text", "text": prompt},
        ]
        r = self._post(self._payload(content, self.cfg.max_new_tokens))
        choice = r["choices"][0]
        raw = choice["message"].get("content") or ""
        usage = r.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        base = self._baseline_tokens(prompt)
        image_tokens = prompt_tokens - base if (prompt_tokens is not None and base is not None) else None
        elapsed = time.time() - t0
        if self.after_read is not None:
            self.after_read()
        return ReadResult(
            raw=raw,
            output_tokens=int(usage.get("completion_tokens") or 0),
            image_soft_tokens=image_tokens,
            elapsed_s=elapsed,
            truncated=choice.get("finish_reason") == "length",
        )

    def recover(self) -> None:
        """Restart the server if it died (e.g. out of GPU memory on a huge page)."""
        if not self.server.alive():
            self.log("llama-server stopped; restarting it.\n" + self.server.tail(10))
            self.server.start()
