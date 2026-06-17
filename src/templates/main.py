"""Cloud Run HTTP handler for lamia scheduled script execution.

Receives POST from Cloud Scheduler, runs the .lm script, returns status.
Logs stdout/stderr to Cloud Logging for observability.
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
            ["lamia", SCRIPT_NAME, "--verbose"],
            capture_output=True,
            text=True,
            cwd="/app/project",
        )

        if result.stdout:
            print(result.stdout[-3000:], flush=True)
        if result.stderr:
            print(result.stderr[-3000:], file=sys.stderr, flush=True)

        status = 200 if result.returncode == 0 else 500
        if status == 500:
            print(f"[lamia] FAILED (exit={result.returncode}): {SCRIPT_NAME}",
                  file=sys.stderr, flush=True)

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
