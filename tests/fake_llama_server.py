"""A stand-in for llama-server used by the tests.

Answers /health (503 for the first second, like a model loading) and
/v1/chat/completions with a canned reply in the format the real prompt asks
for. Accepts and ignores llama-server's other flags. Writes each request body
to --record-dir so tests can inspect what was sent.

Streams like llama-server when asked. Two flags imitate the failures seen on
real pages:
  --loop N     the first N image requests without DRY settings loop on "| "
  --long N     the first N image requests stop at max_tokens without looping
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
ap.add_argument("--loop", type=int, default=0)
ap.add_argument("--long", type=int, default=0)
args, _ = ap.parse_known_args()
if args.version:
    print("fake llama-server")
    raise SystemExit(0)
if args.list_devices:
    print("Available devices:\n  CUDA0: Tesla T4 (15095 MiB, 14992 MiB free)")
    raise SystemExit(0)
STARTED = time.time()
COUNTER = {"n": 0, "loops": 0, "longs": 0}


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
        max_tokens = body.get("max_tokens") or 100000
        finish = "stop"
        if max_tokens == 1:
            pieces = ["x"]
        elif has_image and "dry_multiplier" not in body and COUNTER["loops"] < args.loop:
            COUNTER["loops"] += 1
            pieces = ["<meta>{}</meta>\n<text>\nHeader | Amount\n"] + ["| "] * (max_tokens - 1)
            finish = "length"
        elif has_image and COUNTER["longs"] < args.long:
            COUNTER["longs"] += 1
            pieces = ["<meta>{}</meta>\n<text>\n"] + [f"line {i}\n" for i in range(max_tokens - 1)]
            finish = "length"
        else:
            meta = {"doc_title": f"Fake {COUNTER['n']}", "page_number_printed": "2 of 3"}
            pieces = [f"<|channel>thought\n<channel|><meta>{json.dumps(meta)}</meta>\n",
                      "<text>\nHello\n", "World\n</text>"]
            pieces += [""] * 39   # 42 tokens in all
        n = len(pieces)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n}
        if not body.get("stream"):
            return self._json(200, {
                "choices": [{"message": {"role": "assistant", "content": "".join(pieces)},
                             "finish_reason": finish}],
                "usage": usage,
                "timings": {"prompt_ms": 10.0, "predicted_ms": 20.0},
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for k, piece in enumerate(pieces):
                last = k == n - 1
                chunk = {"choices": [{"delta": {"content": piece}, "index": 0,
                                      "finish_reason": finish if last else None}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass   # the client stopped reading, as the reader does on a loop


HTTPServer(("127.0.0.1", args.port), H).serve_forever()
