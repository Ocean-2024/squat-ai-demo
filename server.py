from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import sys


class DemoHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".task": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def run(port: int) -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = partial(DemoHandler, directory=os.getcwd())
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Squat Coach demo running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)
