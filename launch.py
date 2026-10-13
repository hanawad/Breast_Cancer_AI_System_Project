"""
One-command launcher: starts the FastAPI backend and the Streamlit dashboard.

Usage (from the project root):
    python launch.py
    python launch.py --api-port 8000 --ui-port 8501

"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ["fastapi", "uvicorn", "multipart", "httpx", "torch", "torchvision", "PIL",
            "sklearn", "joblib", "pandas", "numpy", "streamlit", "plotly", "requests"]


def check_dependencies() -> None:
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Missing Python packages: {', '.join(missing)}")
        print(f"Install them with:  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)


def wait_for_backend(url: str, proc: subprocess.Popen, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch backend + dashboard")
    parser.add_argument("--api-port", type=int, default=8001)
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    check_dependencies()

    import socket
    def port_free(p: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return True
        except OSError:
            return False

    if not port_free(args.api_port):
        for candidate in [8001, 8080, 8002, 8003, 8004]:
            if port_free(candidate):
                print(f"Port {args.api_port} is not accessible (WinError 10013 / in use). Using port {candidate} instead.")
                args.api_port = candidate
                break

    api_url = f"http://localhost:{args.api_port}"

    env = os.environ.copy()
    env["BC_API_URL"] = api_url
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print("Starting backend...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(args.api_port)],
        cwd=ROOT, env=env,
    )
    if not wait_for_backend(api_url, backend):
        print("Backend failed to start. See the log above.")
        backend.terminate()
        sys.exit(1)
    print(f"Backend ready: {api_url}/docs")

    print("Starting dashboard...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", str(args.ui_port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env,
    )
    print(f"Dashboard: http://localhost:{args.ui_port}   (Ctrl+C to stop)")

    try:
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (frontend, backend):
            if p.poll() is None:
                p.terminate()
        for p in (frontend, backend):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Stopped.")


if __name__ == "__main__":
    main()
