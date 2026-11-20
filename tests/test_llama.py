"""llama-server plumbing: runtime checks, prebuilt download, GPU code cache,
model downloads, server args, HTTP reader.

The real server needs a GPU; these use fake release tarballs and a fake server process
(tests/fake_llama_server.py) that speaks the same HTTP API.
"""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from pile_scan import llama
from pile_scan.config import ScanConfig
from pile_scan.scanner import run_scan

FAKE_SERVER = Path(__file__).with_name("fake_llama_server.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- server arguments ---------------------------------------------------------

def test_server_args_have_gemma4_image_settings(tmp_path):
    cfg = ScanConfig(max_soft_tokens=1120)
    args = llama.server_args(Path("/bin/llama-server"), Path("m.gguf"), Path("p.gguf"), cfg)

    def val(flag):
        return args[args.index(flag) + 1]

    assert val("--image-min-tokens") == val("--image-max-tokens") == "1120"
    assert int(val("-b")) >= 2048 and int(val("-ub")) >= 2048
    assert int(val("-ub")) >= cfg.max_soft_tokens
    assert val("--mmproj") == "p.gguf" and val("-m") == "m.gguf"
    assert val("--reasoning-budget") == "0"
    assert val("--host") == "127.0.0.1"


def test_model_file_follows_quantization():
    assert ScanConfig(quantization="4bit").model_file == "gemma-4-12b-it-UD-Q4_K_XL.gguf"
    assert ScanConfig(quantization="8bit").model_file == "gemma-4-12b-it-Q8_0.gguf"
    assert ScanConfig(gguf_file="custom.gguf").model_label.endswith("/custom.gguf")


def test_ctx_size_must_fit_page():
    with pytest.raises(ValueError):
        ScanConfig(ctx_size=2048).validate()


def test_release_assets_are_cuda_server_and_runtime():
    urls = llama.release_assets("b12359", "12.8")
    base = "https://github.com/ggml-org/llama.cpp/releases/download/b12359/"
    assert urls == [base + "llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
                    base + "cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz"]


# --- checking the runtime ---------------------------------------------------------

def smi(line):
    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=line + "\n", stderr="")
    return runner


T4 = "Tesla T4, 580.82.07, 7.5, 15360"
L4 = "NVIDIA L4, 580.82.07, 8.9, 23034"


def test_environment_ok_on_current_runtime():
    gpu = llama.check_environment(ScanConfig(), runner=smi(T4), glibc=(2, 39), log=lambda *_: None)
    assert gpu.name == "Tesla T4" and gpu.compute_cap == "7.5" and gpu.needs_jit
    gpu = llama.check_environment(ScanConfig(), runner=smi(L4), glibc=(2, 39), log=lambda *_: None)
    assert not gpu.needs_jit


def test_old_runtime_fails_with_fix():
    with pytest.raises(llama.EnvironmentProblem, match="Runtime version"):
        llama.check_environment(ScanConfig(), runner=smi(T4), glibc=(2, 35))


def test_no_gpu_fails_with_fix():
    def runner(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    with pytest.raises(llama.EnvironmentProblem, match="pick a GPU"):
        llama.check_environment(ScanConfig(), runner=runner, glibc=(2, 39))


def test_old_driver_fails():
    with pytest.raises(llama.EnvironmentProblem, match="driver 535"):
        llama.check_environment(ScanConfig(), runner=smi("Tesla T4, 535.104.05, 7.5, 15360"), glibc=(2, 39))


def test_8bit_on_t4_warns():
    msgs = []
    llama.check_environment(ScanConfig(quantization="8bit"), runner=smi(T4), glibc=(2, 39), log=msgs.append)
    assert any("may not fit" in m for m in msgs)
    msgs.clear()
    llama.check_environment(ScanConfig(), runner=smi(T4), glibc=(2, 39), log=msgs.append)
    assert not any("may not fit" in m for m in msgs)


# --- downloading the prebuilt server ------------------------------------------------

def make_release_tar(path: Path, top: str, files: dict[str, bytes], links: dict[str, str] = {}):
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mode = 0o755 if name == "llama-server" else 0o644
            tf.addfile(info, io.BytesIO(data))
        for name, target in links.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)


def test_prebuilt_download_unpacks_flat_and_is_reused(tmp_path):
    fetched = []

    def fake_download(url, dest, *, token=None, log=print):
        fetched.append(url)
        name = url.rsplit("/", 1)[1]
        if name.startswith("cudart"):
            make_release_tar(dest, name.removesuffix(".tar.gz"), {"libcudart.so.12.8.90": b"rt"},
                             {"libcudart.so.12": "libcudart.so.12.8.90"})
        else:
            make_release_tar(dest, "llama-b12359", {"llama-server": b"#!/bin/sh\n",
                                                     "libggml-cuda.so": b"cuda"})

    cfg = ScanConfig(local_dir=str(tmp_path / "vm"))
    server = llama.ensure_server_binary(cfg, download=fake_download, log=lambda *_: None)
    assert len(fetched) == 2
    folder = server.parent
    assert server.name == "llama-server" and server.stat().st_mode & 0o111
    # Libraries sit next to the server, where its RUNPATH ($ORIGIN) looks.
    assert (folder / "libggml-cuda.so").read_bytes() == b"cuda"
    assert (folder / "libcudart.so.12").read_bytes() == b"rt"
    assert not list((tmp_path / "vm" / "downloads").iterdir())   # tarballs removed

    llama.ensure_server_binary(cfg, download=fake_download, log=lambda *_: None)
    assert len(fetched) == 2


def test_half_unpacked_server_is_redownloaded(tmp_path):
    cfg = ScanConfig(local_dir=str(tmp_path / "vm"))
    folder = tmp_path / "vm" / f"llama-{cfg.llama_cpp_build}-cuda-{cfg.llama_cpp_cuda}"
    folder.mkdir(parents=True)
    (folder / "llama-server").write_text("partial")   # no .complete marker
    calls = []

    def fake_download(url, dest, *, token=None, log=print):
        calls.append(url)
        make_release_tar(dest, "x", {"llama-server": b"ok"})

    llama.ensure_server_binary(cfg, download=fake_download, log=lambda *_: None)
    assert len(calls) == 2 and (folder / "llama-server").read_bytes() == b"ok"


def script(tmp_path, body):
    path = tmp_path / "llama-server"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_verify_server_sees_cuda(tmp_path):
    binary = script(tmp_path, f'exec {sys.executable} {FAKE_SERVER} "$@"')
    assert llama.verify_server(binary).startswith("CUDA0: Tesla T4")


def test_verify_server_fails_when_cuda_backend_missing(tmp_path):
    binary = script(tmp_path, 'echo "Available devices:"; echo "  (none)"')
    with pytest.raises(llama.EnvironmentProblem, match="can't see the GPU"):
        llama.verify_server(binary)


def test_verify_server_explains_glibc_error(tmp_path):
    binary = script(tmp_path, "echo \"llama-server: /lib/x86_64-linux-gnu/libc.so.6: version "
                              "\\`GLIBC_2.38' not found\" >&2; exit 1")
    with pytest.raises(llama.EnvironmentProblem, match="newer glibc"):
        llama.verify_server(binary)


# --- GPU code cache on Drive ---------------------------------------------------------

def test_jit_cache_round_trip(tmp_path):
    gpu = llama.GpuInfo("Tesla T4", "580.82.07", "7.5", 15360)
    cfg = ScanConfig(local_dir=str(tmp_path / "vm1"), cache_dir=str(tmp_path / "drive"))
    jit = llama.JitCache(cfg, gpu, log=lambda *_: None)
    assert jit.restore() is False
    env = jit.env()
    assert env["CUDA_CACHE_PATH"] == str(jit.local)
    assert env["CUDA_CACHE_MAXSIZE"] == str(4 * 1024 ** 3)
    assert jit.save() is False                        # nothing compiled yet
    (jit.local / "ComputeCache" / "a").mkdir(parents=True)
    (jit.local / "ComputeCache" / "index").write_bytes(b"idx")
    (jit.local / "ComputeCache" / "a" / "k1").write_bytes(b"kernel")
    assert jit.save() is True
    assert jit.save() is False                        # unchanged: no second upload
    assert "Tesla-T4-580.82.07" in jit.drive_file.name and cfg.llama_cpp_build in jit.drive_file.name

    cfg2 = ScanConfig(local_dir=str(tmp_path / "vm2"), cache_dir=cfg.cache_dir)
    jit2 = llama.JitCache(cfg2, gpu, log=lambda *_: None)
    assert jit2.restore() is True
    assert (jit2.local / "ComputeCache" / "a" / "k1").read_bytes() == b"kernel"
    assert jit2.save() is False                       # restored copy needs no upload

    # A different driver gets its own cache file.
    other = llama.JitCache(cfg2, llama.GpuInfo("Tesla T4", "590.10", "7.5", 15360), log=lambda *_: None)
    assert other.drive_file != jit.drive_file


def test_corrupt_jit_cache_is_ignored(tmp_path):
    gpu = llama.GpuInfo("Tesla T4", "580.82.07", "7.5", 15360)
    cfg = ScanConfig(local_dir=str(tmp_path / "vm"), cache_dir=str(tmp_path / "drive"))
    jit = llama.JitCache(cfg, gpu, log=lambda *_: None)
    jit.drive_file.parent.mkdir(parents=True)
    jit.drive_file.write_bytes(b"not a tar")
    assert jit.restore() is False


def test_server_gets_cache_environment(tmp_path):
    args = [sys.executable, "-c", "import os, sys; print('cache=' + os.environ['CUDA_CACHE_PATH']); sys.exit(1)"]
    server = llama.LlamaServer(args, port=free_port(), log_path=tmp_path / "s.log", log=lambda *_: None,
                               env={**__import__("os").environ, "CUDA_CACHE_PATH": "/x/jit"})
    with pytest.raises(RuntimeError, match="cache=/x/jit"):
        server.start(timeout=10)


# --- model files ----------------------------------------------------------------

def test_model_download_and_drive_cache(tmp_path):
    fetched = []

    def fake_download(url, dest, *, token=None, log=print):
        fetched.append(url)
        dest.write_bytes(b"gguf")

    cfg = ScanConfig(model_dir=str(tmp_path / "vm1"), cache_dir=str(tmp_path / "drive"),
                     cache_models_on_drive=True)
    model, mmproj = llama.ensure_model_files(cfg, download=fake_download, log=lambda *_: None)
    assert model.name == "gemma-4-12b-it-UD-Q4_K_XL.gguf" and mmproj.name == "mmproj-F16.gguf"
    assert fetched[0] == ("https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/"
                          "gemma-4-12b-it-UD-Q4_K_XL.gguf")
    # Second VM: comes from Drive, no download.
    cfg2 = ScanConfig(model_dir=str(tmp_path / "vm2"), cache_dir=cfg.cache_dir, cache_models_on_drive=True)
    llama.ensure_model_files(cfg2, download=fake_download, log=lambda *_: None)
    assert len(fetched) == 2
    # Same VM again: nothing to do.
    llama.ensure_model_files(cfg2, download=fake_download, log=lambda *_: None)
    assert len(fetched) == 2


def test_download_resumes_partial_file(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    data = bytes(range(256)) * 100

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dest = tmp_path / "f.gguf"
    (tmp_path / "f.gguf.part").write_bytes(data[:1000])
    llama._download(f"http://127.0.0.1:{srv.server_port}/f", dest, log=lambda *_: None)
    srv.shutdown()
    assert dest.read_bytes() == data
    assert not (tmp_path / "f.gguf.part").exists()


# --- talking to the server ----------------------------------------------------------

@pytest.fixture
def fake_server(tmp_path):
    port = free_port()
    rec = tmp_path / "requests"
    rec.mkdir()
    args = [sys.executable, str(FAKE_SERVER), "--port", str(port), "--record-dir", str(rec)]
    server = llama.LlamaServer(args, port=port, log_path=tmp_path / "server.log", log=lambda *_: None)
    server.start(timeout=30)
    yield server, port, rec
    server.stop()


def test_reader_sends_image_first_and_counts_image_tokens(fake_server, tmp_path):
    from PIL import Image

    server, port, rec = fake_server
    cfg = ScanConfig(server_port=port)
    reader = llama.LlamaPageReader(server, cfg, log=lambda *_: None)
    res = reader.read(Image.new("RGB", (64, 64), "white"), "Read this page.")
    assert "<text>" in res.raw
    assert res.output_tokens == 42
    assert res.image_soft_tokens == 1080  # 1680 with image - 600 text only
    assert res.truncated is False

    sent = json.loads(sorted(rec.glob("req*.json"))[0].read_text())
    content = sent["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "Read this page."}
    assert sent["temperature"] == 0
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["max_tokens"] == cfg.max_new_tokens


def test_after_read_hook_runs_per_page(fake_server):
    from PIL import Image

    server, port, _ = fake_server
    calls = []
    reader = llama.LlamaPageReader(server, ScanConfig(server_port=port), log=lambda *_: None,
                                   after_read=lambda: calls.append(1))
    reader.read(Image.new("RGB", (8, 8)), "p")
    reader.read(Image.new("RGB", (8, 8)), "p")
    assert len(calls) == 2


def test_recover_restarts_dead_server(fake_server):
    server, port, _ = fake_server
    server.proc.kill()
    server.proc.wait()
    assert not server.alive()
    reader = llama.LlamaPageReader(server, ScanConfig(server_port=port), log=lambda *_: None)
    reader.recover()
    assert server.alive()


def test_server_that_exits_reports_its_log(tmp_path):
    args = [sys.executable, "-c", "import sys; print('CUDA error: out of memory'); sys.exit(1)"]
    server = llama.LlamaServer(args, port=free_port(), log_path=tmp_path / "s.log", log=lambda *_: None)
    with pytest.raises(RuntimeError, match="out of memory"):
        server.start(timeout=10)


def test_full_scan_through_fake_server(fake_server, cfg):
    server, port, _ = fake_server
    cfg.server_port = port
    reader = llama.LlamaPageReader(server, cfg, log=lambda *_: None)
    s = run_scan(cfg, reader, log=lambda *_: None)
    assert s.pages_read == 7 and s.pages_failed == 0
    import sqlite3

    conn = sqlite3.connect(cfg.drive_db_path)
    rows = conn.execute("SELECT doc_title, page_number_int, page_total_int, image_soft_tokens, ocr_text, "
                        "model_id FROM pages").fetchall()
    assert all(r[3] == 1080 and r[4] == "Hello\nWorld" and (r[1], r[2]) == (2, 3) for r in rows)
    assert rows[0][5] == "unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf"
