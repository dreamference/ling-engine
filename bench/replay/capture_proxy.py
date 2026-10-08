"""Logging reverse proxy: records every request body a client sends to the model server.

Used once in M0 to learn the exact wire format Mightling's agent sends (endpoint, tools array,
sampling parameters, chat-template kwargs), so the replay can rebuild requests from rollouts in
the same shape. Request bodies go to a local JSONL file and never into the repository.

Usage: python3 capture_proxy.py LISTEN_PORT UPSTREAM_URL OUT.jsonl
"""
import http.client
import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
UP = urllib.parse.urlparse(sys.argv[2])
OUT = sys.argv[3]
LOCK = threading.Lock()


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if body:
            with LOCK, open(OUT, "a") as f:
                f.write(json.dumps({"t": time.time(), "method": method, "path": self.path,
                                    "headers": dict(self.headers),
                                    "body": body.decode("utf-8", "replace")}) + "\n")
        conn = http.client.HTTPConnection(UP.hostname, UP.port, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        conn.request(method, self.path, body=body or None, headers=headers)
        r = conn.getresponse()
        self.send_response(r.status)
        chunked = r.getheader("Transfer-Encoding", "").lower() == "chunked" or r.getheader("Content-Length") is None
        for k, v in r.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            data = r.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        conn.close()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Proxy).serve_forever()
