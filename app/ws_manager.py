"""Thread-safe WebSocket Connection Manager and live hardware telemetry broadcaster."""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import psutil
from fastapi import WebSocket


def _query_nvidia_smi() -> dict[str, Any] | None:
    """Read NVIDIA GPU utilization/memory/name via nvidia-smi without requiring CUDA torch."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        ).strip()
        if not out:
            return None
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "vram_used_mb": float(parts[1]),
                "vram_total_mb": float(parts[2]),
                "gpu_util_pct": float(parts[3]),
                "temp_c": float(parts[4]),
            }
    except Exception:
        return None
    return None


def collect_system_telemetry(torch_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot current CPU, RAM, disk, and GPU stats."""
    vm = psutil.virtual_memory()
    gpu = _query_nvidia_smi()
    info = torch_info or {}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()}",
        "os_type": info.get("os_type") or ("windows" if platform.system().lower().startswith("win") else ("macos" if platform.system().lower() == "darwin" else "linux")),
        "python": platform.python_version(),
        "target_python": info.get("target_python", "3.11.6"),
        "python_compatible": info.get("python_compatible", True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_percent": vm.percent,
        "gpu": gpu,
        "torch_version": info.get("torch_version", "unknown"),
        "torch_cuda_version": info.get("torch_cuda_version"),
        "torch_cuda_available": info.get("cuda_available", False),
        "cuda_toolkit_version": info.get("cuda_toolkit_version"),
        "target_cuda": info.get("target_cuda", "11.8"),
        "cuda_version_mismatch": bool(info.get("cuda_version_mismatch", False)),
        "cuda_install_cmd": info.get("cuda_install_cmd", ""),
        "setup_cmd": info.get("setup_cmd", ""),
        "ultralytics_version": info.get("ultralytics_version", "unknown"),
        "cuda_advisory": bool(gpu and not info.get("cuda_available", False)),
    }


class WebSocketManager:
    """Manages active WebSocket clients and supports thread-safe event broadcasts."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._recent_logs: list[dict[str, Any]] = []
        self._max_logs = 400
        self.torch_info: dict[str, Any] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        await self.send_personal(
            websocket,
            {
                "type": "connected",
                "data": {
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "clients": len(self._connections),
                    "recent_logs": self._recent_logs[-80:],
                },
            },
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def send_personal(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            await self.disconnect(websocket)

    async def broadcast(self, event_type: str, data: Any) -> None:
        message = {
            "type": event_type,
            "ts": time.time(),
            "data": data,
        }
        if event_type == "log":
            self._recent_logs.append(message)
            if len(self._recent_logs) > self._max_logs:
                self._recent_logs = self._recent_logs[-self._max_logs :]

        raw = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            targets = list(self._connections)

        stale: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(raw)
            except Exception:
                stale.append(ws)
        if stale:
            async with self._lock:
                for ws in stale:
                    self._connections.discard(ws)

    def broadcast_sync(self, event_type: str, data: Any) -> None:
        """Schedule a broadcast safely from background worker threads."""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(event_type, data), self._loop)
        except Exception:
            pass

    def log_sync(self, text: str, level: str = "info", source: str = "system") -> None:
        self.broadcast_sync(
            "log",
            {
                "level": level,
                "source": source,
                "message": text,
                "time": datetime.now().strftime("%H:%M:%S"),
            },
        )

    async def telemetry_loop(self) -> None:
        """Periodically push live CPU/RAM/GPU metrics to connected WebSocket clients."""
        while True:
            try:
                if self._connections:
                    stats = await asyncio.to_thread(collect_system_telemetry, self.torch_info)
                    await self.broadcast("telemetry", stats)
            except asyncio.CancelledError:
                break
            except Exception:
                pass
            await asyncio.sleep(2.5)


ws_manager = WebSocketManager()
