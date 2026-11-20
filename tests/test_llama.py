"""llama-server plumbing: build caching, model downloads, server args, HTTP reader.

The real server needs a GPU; these use a fake build and a fake server process
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


def test_build_commands_target_one_arch_static():
    cfg_cmd, build_cmd = llama.build_commands(Path("/src"), Path("/src/build"), "75", 2)
    assert "-DGGML_CUDA=ON" in cfg_cmd
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in cfg_cmd
    assert "-DBUILD_SHARED_LIBS=OFF" in cfg_cmd
    assert build_cmd[-2:] == ["--target", "llama-server"]


def test_cuda_arch_parses_nvidia_smi():
    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="7.5\n", stderr="")

    assert llama.cuda_arch(runner) == "75"


# --- build caching ----------------------------------------------------------------

class FakeBuild:
    """Records commands; 'builds' llama-server as a copy of a tiny script."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[0] == "git" and cmd[1] == "clone":
            Path(cmd[-1]).mkdir(parents=True)
        if cmd[0] == "cmake" and "--build" in cmd:
            out = Path(cmd[cmd.index("--build") + 1]) / "bin" / "llama-server"
            out.parent.mkdir(parents=True)
            out.write_text("#!/bin/sh\necho fake\n")
        if cmd[0] == "git" and "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[-1] == "--version":
            subprocess.run(cmd, check=True, capture_output=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    @property
    def builds(self):
        return sum(1 for c in self.calls if c[0] == "cmake" and "--build" in c)


def test_build_once_then_reuse_cache(tmp_path):
    cfg = ScanConfig(cache_dir=str(tmp_path / "drive" / "cache"), local_dir=str(tmp_path / "vm1"))
    fb = FakeBuild()
    b1 = llama.ensure_server_binary(cfg, runner=fb, arch="75", log=lambda *_: None)
    assert fb.builds == 1 and b1.exists()
    entry = llama.cache_entry(cfg.cache_dir, cfg.llama_cpp_ref, "75")
    info = json.loads((entry / "build-info.json").read_text())
    assert info["commit"] == "abc123" and info["cuda_arch"] == "75"
    assert not list(entry.glob("*.tmp-*"))

    # A fresh VM (empty local disk) reuses the Drive copy without building.
    cfg2 = ScanConfig(cache_dir=cfg.cache_dir, local_dir=str(tmp_path / "vm2"))
    b2 = llama.ensure_server_binary(cfg2, runner=fb, arch="75", log=lambda *_: None)
    assert fb.builds == 1
    assert b2.stat().st_mode & 0o111  # executable bit restored

    # A different GPU type gets its own build.
    llama.ensure_server_binary(cfg2, runner=fb, arch="89", log=lambda *_: None)
    assert fb.builds == 2


def test_broken_cached_binary_is_rebuilt(tmp_path):
    cfg = ScanConfig(cache_dir=str(tmp_path / "cache"), local_dir=str(tmp_path / "vm"))
    entry = llama.cache_entry(cfg.cache_dir, cfg.llama_cpp_ref, "75")
    entry.mkdir(parents=True)
    (entry / "llama-server").write_text("not a program")
    fb = FakeBuild()
    llama.ensure_server_binary(cfg, runner=fb, arch="75", log=lambda *_: None)
    assert fb.builds == 1


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
