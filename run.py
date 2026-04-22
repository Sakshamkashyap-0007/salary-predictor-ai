"""
run.py — Single-command launcher for AI Salary Predictor.

Usage:
    python run.py

This script:
  1. Checks for model + data; runs train.py if model is missing.
  2. Kills anything already on port 8000 AND the frontend port (Windows + Unix safe).
  3. Starts the FastAPI backend on port 8000 (subprocess).
  4. Serves index.html / styles.css / script.js directly from the
     project root — no 'frontend' subfolder required.
  5. Opens the browser automatically.
  6. Shuts both servers cleanly on Ctrl+C.
"""

import http.server
import os
import platform
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "salary_model.pkl")
DATA_PATH  = os.path.join(ROOT, "data",   "salary_data.csv")

API_PORT           = 8000
FRONTEND_PORT_PREF = 5500          # preferred; falls back if occupied
FRONTEND_PORT_RANGE = range(5500, 5510)  # try up to 5509

# ── Where are the HTML/CSS/JS files? ────────────────────────────────────────
_frontend_subdir = os.path.join(ROOT, "frontend")
FRONTEND_DIR = _frontend_subdir if os.path.isdir(_frontend_subdir) else ROOT


# ── Port utilities ───────────────────────────────────────────────────────────

def _is_port_free(port: int) -> bool:
    """Return True when nothing is listening on 127.0.0.1:<port>."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _free_port(port: int) -> None:
    """Kill whatever process is holding the given port (Windows + Unix)."""
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True,
                stderr=subprocess.DEVNULL,
            ).decode()
            pids = set()
            for line in out.strip().splitlines():
                parts = line.split()
                # Only match lines where the local address ends with :<port>
                if len(parts) >= 2 and parts[1].endswith(f":{port}") and parts[-1].isdigit():
                    pids.add(parts[-1])
            for pid in pids:
                subprocess.call(
                    f"taskkill /F /PID {pid}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if pids:
                print(f"[run.py] Freed port {port} (PID {', '.join(pids)})")
                time.sleep(0.8)
        except subprocess.CalledProcessError:
            pass
    else:
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"], stderr=subprocess.DEVNULL,
            ).decode().strip()
            if out:
                for pid in out.splitlines():
                    os.kill(int(pid), signal.SIGTERM)
                print(f"[run.py] Freed port {port}")
                time.sleep(0.5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


def _pick_frontend_port() -> int:
    """
    Try to free the preferred port first.
    If it is still busy after freeing (e.g. Windows firewall blocks 5500),
    walk through FRONTEND_PORT_RANGE until a free port is found.
    """
    _free_port(FRONTEND_PORT_PREF)
    time.sleep(0.3)

    for port in FRONTEND_PORT_RANGE:
        if _is_port_free(port):
            return port

    # Last-resort: let the OS pick any free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Setup helpers ────────────────────────────────────────────────────────────

def _ensure_data() -> None:
    if not os.path.exists(DATA_PATH):
        print("[run.py] Dataset not found — generating...")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "generate_data.py")],
            check=True, cwd=ROOT,
        )


def _ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print("[run.py] Model not found — training (this takes ~30 s)...")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "train.py")],
            check=True, cwd=ROOT,
        )


def _start_api() -> subprocess.Popen:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("[run.py] ERROR: uvicorn not found.")
        print("         Run:  pip install fastapi uvicorn")
        sys.exit(1)

    _free_port(API_PORT)
    time.sleep(0.3)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api:app",
        "--host", "127.0.0.1",
        "--port", str(API_PORT),
        "--log-level", "warning",
    ]
    return subprocess.Popen(cmd, cwd=ROOT)


def _start_frontend(port: int) -> socketserver.TCPServer:
    """Serve FRONTEND_DIR over plain HTTP on the given port (background thread)."""
    os.chdir(FRONTEND_DIR)

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", port), _QuietHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 58)
    print("   AI Salary Predictor — starting up")
    print("=" * 58)

    _ensure_data()
    _ensure_model()

    print(f"[run.py] Starting FastAPI   → http://127.0.0.1:{API_PORT}")
    api_proc = _start_api()

    frontend_port = _pick_frontend_port()
    print(f"[run.py] Serving frontend   → http://127.0.0.1:{frontend_port}")
    print(f"[run.py] Serving files from → {FRONTEND_DIR}")

    try:
        _start_frontend(frontend_port)
    except OSError as exc:
        print(f"[run.py] ERROR: Could not bind frontend server — {exc}")
        print("[run.py] Try closing VS Code Live Server or any app using port 5500-5509.")
        api_proc.terminate()
        sys.exit(1)

    # Let uvicorn finish binding before opening the browser
    time.sleep(1.8)

    url = f"http://127.0.0.1:{frontend_port}/index.html"
    print(f"[run.py] Opening browser    → {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print("[run.py] Press Ctrl+C to stop.")
    print("=" * 58)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    def _shutdown(sig=None, frame=None):
        print("\n[run.py] Shutting down...")
        api_proc.terminate()
        try:
            api_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (OSError, ValueError):
        pass

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()