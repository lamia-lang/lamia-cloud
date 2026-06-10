"""Cloud Run HTTP handler for lamia scheduled script execution.

Receives POST from Cloud Scheduler, runs the .lm script, returns status.
"""

import json
import os
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler


SCRIPT_NAME = os.environ.get("LAMIA_SCRIPT", "script.lm")
PORT = int(os.environ.get("PORT", "8080"))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        result = subprocess.run(
            [sys.executable, "-m", "lamia", "--file", SCRIPT_NAME],
            capture_output=True,
            text=True,
            cwd="/app/project",
        )

        status = 200 if result.returncode == 0 else 500
        body = json.dumps({
            "exit_code": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        })

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        print(format % args, flush=True)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Lamia cloud runner listening on port {PORT}", flush=True)
    server.serve_forever()
