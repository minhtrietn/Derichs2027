"""Entrypoint to launch DERICH YOLO Auto-Training Studio (FastAPI + WebSocket)."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn

from app.config import ensure_directories


def _open_browser_later(url: str) -> None:
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="DERICH YOLO Auto-Training Studio (WebSocket Web App)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1, use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on startup")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    ensure_directories()
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print("=" * 64)
    print("  DERICH YOLO Auto-Training Studio (YOLOv8 / YOLO11 / YOLO26)")
    print(f"  Web UI & WebSocket : {url}")
    print(f"  WebSocket Endpoint : ws://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/ws")
    print("=" * 64)

    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
