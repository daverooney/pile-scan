"""Running Gemma 4 through llama.cpp's own server.

Three jobs, each safe to repeat at the start of every Colab session:

1. `ensure_server_binary`: build `llama-server` with CUDA once per GPU type and
   llama.cpp version, and cache the binary on Drive. Later sessions copy it
   back in seconds.
2. `ensure_model_files`: download the GGUF model and its vision projector.
3. `LlamaServer` and `LlamaPageReader`: start the server, then send it one page
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
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .config import ScanConfig
from .reader import ReadResult

LLAMA_CPP_GIT = "https://github.com/ggml-org/llama.cpp"
BATCH_SIZE = 2048

Runner = Callable[..., subprocess.CompletedProcess]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


# --- building and caching the server -----------------------------------------

def cuda_arch(runner: Runner = _run) -> str:
    """Compute capability of GPU 0 as CMake wants it: "75" for a T4, "89" for an L4."""
    out = runner(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                 capture_output=True, text=True).stdout
    first = out.strip().splitlines()[0].strip()
    return first.replace(".", "")


def build_commands(src: Path, build: Path, arch: str, jobs: int) -> list[list[str]]:
    return [
        ["cmake", "-S", str(src), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release",
         "-DGGML_CUDA=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}",   # only this GPU: much faster to compile
         "-DBUILD_SHARED_LIBS=OFF",              # one self-contained binary to cache
         "-DLLAMA_OPENSSL=OFF",                  # we download models ourselves
         "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=OFF"],
        ["cmake", "--build", str(build), "--config", "Release", "-j", str(jobs), "--target", "llama-server"],
    ]


def cache_entry(cache_dir: str | Path, ref: str, arch: str) -> Path:
    return Path(cache_dir) / "llama.cpp" / f"{ref}-sm{arch}"


def _binary_works(binary: Path, runner: Runner) -> bool:
    try:
        runner([str(binary), "--version"], capture_output=True, text=True, timeout=60)
        return True
    except Exception:
        return False


def ensure_server_binary(cfg: ScanConfig, *, rebuild: bool = False, log=print,
                         runner: Runner = _run, arch: str | None = None) -> Path:
    """Return a path to a working llama-server on local disk.

    Reuses the copy cached on Drive when there is one for this llama.cpp
    version and GPU type; otherwise builds it (slow, once) and caches it.
    """
    arch = arch or cuda_arch(runner)
    entry = cache_entry(cfg.cache_dir, cfg.llama_cpp_ref, arch)
    cached = entry / "llama-server"
    local_bin = Path(cfg.local_dir) / "bin" / "llama-server"
    local_bin.parent.mkdir(parents=True, exist_ok=True)

    if cached.exists() and not rebuild:
        shutil.copyfile(cached, local_bin)   # Drive doesn't keep the executable bit
        local_bin.chmod(0o755)
        if _binary_works(local_bin, runner):
            log(f"Using cached llama-server ({cfg.llama_cpp_ref}, sm{arch}) from {entry}")
            return local_bin
        log("Cached llama-server didn't run; rebuilding.")

    src = Path(cfg.local_dir) / "llama.cpp-src"
    build = src / "build"
    if src.exists():
        shutil.rmtree(src)
    # Colab ships the CUDA toolkit in /usr/local/cuda but doesn't always put
    # nvcc on PATH; CMake finds it through CUDACXX.
    if shutil.which("nvcc") is None and Path("/usr/local/cuda/bin/nvcc").exists():
        os.environ.setdefault("CUDACXX", "/usr/local/cuda/bin/nvcc")
    log(f"Building llama-server {cfg.llama_cpp_ref} for sm{arch}. This happens once per GPU type "
        "and llama.cpp version, and takes a while.")
    t0 = time.time()
    runner(["git", "clone", "--depth", "1", "--branch", cfg.llama_cpp_ref, LLAMA_CPP_GIT, str(src)])
    for cmd in build_commands(src, build, arch, os.cpu_count() or 2):
        runner(cmd)
    built = build / "bin" / "llama-server"
    if not built.exists():
        raise RuntimeError(f"Build finished but {built} is missing.")
    commit = runner(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"Built in {(time.time() - t0) / 60:.1f} min.")

    shutil.copyfile(built, local_bin)
    local_bin.chmod(0o755)

    # Cache on Drive: write under a temporary name, then rename, so a session
    # killed mid-copy never leaves a half-written binary that looks complete.
    entry.mkdir(parents=True, exist_ok=True)
    tmp = entry / f"llama-server.tmp-{os.getpid()}"
    shutil.copyfile(built, tmp)
    os.replace(tmp, cached)
    (entry / "build-info.json").write_text(json.dumps({
        "ref": cfg.llama_cpp_ref, "commit": commit, "cuda_arch": arch,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))
    log(f"Cached llama-server at {cached}")
    shutil.rmtree(src, ignore_errors=True)
    return local_bin


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

    def __init__(self, args: list[str], *, port: int, log_path: Path, log=print):
        self.args = args
        self.port = port
        self.log_path = log_path
        self.log = log
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, timeout: float = 900) -> "LlamaServer":
        self.stop()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.args, stdout=self._log_fh, stderr=subprocess.STDOUT)
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

    def __init__(self, server: LlamaServer, cfg: ScanConfig, *, request_timeout: float = 1800, log=print):
        self.server = server
        self.cfg = cfg
        self.timeout = request_timeout
        self.log = log
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
        return ReadResult(
            raw=raw,
            output_tokens=int(usage.get("completion_tokens") or 0),
            image_soft_tokens=image_tokens,
            elapsed_s=time.time() - t0,
            truncated=choice.get("finish_reason") == "length",
        )

    def recover(self) -> None:
        """Restart the server if it died (e.g. out of GPU memory on a huge page)."""
        if not self.server.alive():
            self.log("llama-server stopped; restarting it.\n" + self.server.tail(10))
            self.server.start()
