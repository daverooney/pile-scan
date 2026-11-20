"""A stand-in for llama-server used by the tests.

Answers /health (503 for the first second, like a model loading) and
/v1/chat/completions with a canned reply in the format the real prompt asks
for. Accepts and ignores llama-server's other flags. Writes each request body
to --record-dir so tests can inspect what was sent.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int)
ap.add_argument("--record-dir")
ap.add_argument("--version", action="store_true")
ap.add_argument("--list-devices", action="store_true")
args, _ = ap.parse_known_args()
if args.version:
    print("fake llama-server")
    raise SystemExit(0)
if args.list_devices:
    print("Available devices:\n  CUDA0: Tesla T4 (15095 MiB, 14992 MiB free)")
    raise SystemExit(0)
STARTED = time.time()
COUNTER = {"n": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if time.time() - STARTED < 1:
                return self._json(503, {"error": {"code": 503, "message": "Loading model"}})
            return self._json(200, {"status": "ok"})
        self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        COUNTER["n"] += 1
        if args.record_dir:
            Path(args.record_dir, f"req{COUNTER['n']:03d}.json").write_text(json.dumps(body))
        content = body["messages"][0]["content"]
        has_image = any(c["type"] == "image_url" for c in content)
        prompt_tokens = 600 + (1080 if has_image else 0)
        if body.get("max_tokens") == 1:
            text, n = "x", 1
        else:
            meta = {"doc_title": f"Fake {COUNTER['n']}", "page_number_printed": "2 of 3"}
            text = (f"<|channel>thought\n<channel|><meta>{json.dumps(meta)}</meta>\n"
                    f"<text>\nHello\nWorld\n</text>")
            n = 42
        self._json(200, {
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n},
            "timings": {"prompt_ms": 10.0, "predicted_ms": 20.0},
        })


HTTPServer(("127.0.0.1", args.port), H).serve_forever()
