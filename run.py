#!/usr/bin/env python3
"""Unified cross-platform auto-launcher & environment setup for DERICH YOLO Studio.

Automatically detects Windows, Linux, or macOS:
- Selects or creates an OS-compatible virtualenv (.venv, or .venv_linux/.venv_win/.venv_macos
  when the same folder is shared across operating systems).
- Detects NVIDIA GPU & local CUDA Toolkit (preferring CUDA 11.8) and configures PATH / LD_LIBRARY_PATH
  before launching the server.
- Installs missing dependencies automatically (or on demand with --setup).
- Starts main.py cleanly on both Windows and Linux (including headless Linux servers).

Usage:
  Windows: python run.py              (or double-click start_app.bat)
  Linux  : python3 run.py             (or ./start_app.sh)
  Setup  : python3 run.py --setup     (add --cpu for CPU-only, --system to skip venv)
  Check  : python3 run.py --check
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
MAIN_SCRIPT = BASE_DIR / "main.py"
PYTORCH_CU118_INDEX = "https://download.pytorch.org/whl/cu118"
PYTORCH_CU118_PKGS = ["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"]
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def detect_os() -> str:
    sys_name = platform.system().lower()
    if sys_name.startswith("win") or os.name == "nt":
        return "windows"
    if sys_name == "darwin":
        return "macos"
    return "linux"


def venv_python_path(venv_dir: Path, os_type: str) -> Path:
    if os_type == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def resolve_venv_dir(os_type: str) -> Path:
    """Use .venv if it belongs to the current OS (or doesn't exist yet); otherwise isolate by OS."""
    default_venv = BASE_DIR / ".venv"
    if not default_venv.exists():
        return default_venv
    if venv_python_path(default_venv, os_type).exists():
        return default_venv
    # .venv belongs to another OS (e.g. Windows .venv copied onto Linux) — keep both safely separated!
    suffix = {"windows": "_win", "linux": "_linux", "macos": "_macos"}[os_type]
    return BASE_DIR / f".venv{suffix}"


def _probe_python_version(cmd: list[str]) -> tuple[int, int, int] | None:
    try:
        out = subprocess.check_output(
            [*cmd, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
        parts = out.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def find_base_python(os_type: str) -> list[str]:
    """Find best base Python interpreter (preferring Python 3.11)."""
    candidates: list[list[str]] = []
    if os_type == "windows":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            p311 = Path(local_app) / "Programs" / "Python" / "Python311" / "python.exe"
            if p311.exists():
                candidates.append([str(p311)])
        if shutil.which("py"):
            candidates.append(["py", "-3.11"])
        candidates.append([sys.executable])
        if shutil.which("python"):
            candidates.append(["python"])
    else:
        for name in ("python3.11", "python3", "python"):
            found = shutil.which(name)
            if found:
                candidates.append([found])
        candidates.append([sys.executable])

    fallback: list[str] | None = None
    seen: set[tuple[str, ...]] = set()
    for cmd in candidates:
        key = tuple(cmd)
        if key in seen:
            continue
        seen.add(key)
        ver = _probe_python_version(cmd)
        if ver is None:
            continue
        if ver[:2] == (3, 11):
            return cmd
        if fallback is None and ver >= (3, 10):
            fallback = cmd
    return fallback or [sys.executable]


def detect_nvidia_gpu() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        ).strip()
        return out.splitlines()[0].strip() if out else None
    except Exception:
        return None


def detect_cuda_root(os_type: str) -> Path | None:
    candidates: list[Path] = []
    if os_type == "windows":
        for k in ("CUDA_PATH_V11_8", "CUDA_PATH"):
            v = os.environ.get(k)
            if v:
                candidates.append(Path(v))
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        candidates.append(base / "v11.8")
        if base.is_dir():
            for child in sorted(base.iterdir(), reverse=True):
                if child.is_dir():
                    candidates.append(child)
    else:
        candidates.append(Path("/usr/local/cuda-11.8"))
        for k in ("CUDA_HOME", "CUDA_PATH"):
            v = os.environ.get(k)
            if v:
                candidates.append(Path(v))
        candidates.append(Path("/usr/local/cuda"))
        usr_local = Path("/usr/local")
        if usr_local.is_dir():
            for child in sorted(usr_local.glob("cuda-*"), reverse=True):
                if child.is_dir():
                    candidates.append(child)

    for p in candidates:
        if p.is_dir():
            return p
    return None


def build_runtime_env(os_type: str) -> dict[str, str]:
    """Configure CUDA, allocator, and Ultralytics paths cleanly for the current OS."""
    env = os.environ.copy()
    cuda_root = detect_cuda_root(os_type)
    if cuda_root is not None:
        env.setdefault("CUDA_PATH", str(cuda_root))
        env.setdefault("CUDA_HOME", str(cuda_root))
        if os_type == "windows":
            extra_bins = [
                str(d)
                for d in (cuda_root / "bin", cuda_root / "libnvvp", cuda_root / "extras" / "CUPTI" / "lib64")
                if d.is_dir()
            ]
            if extra_bins:
                cur = env.get("PATH", "")
                env["PATH"] = os.pathsep.join(extra_bins + ([cur] if cur else []))
        else:
            bin_dir = cuda_root / "bin"
            if bin_dir.is_dir():
                cur = env.get("PATH", "")
                env["PATH"] = f"{bin_dir}{os.pathsep}{cur}" if cur else str(bin_dir)
            extra_libs = [
                str(d)
                for d in (cuda_root / "lib64", cuda_root / "extras" / "CUPTI" / "lib64")
                if d.is_dir()
            ]
            if extra_libs:
                cur_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = os.pathsep.join(extra_libs + ([cur_ld] if cur_ld else []))

    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    env.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "studio_data" / ".ultralytics"))
    return env


def has_core_deps(python_cmd: list[str], env: dict[str, str]) -> bool:
    """Fast dependency check using importlib.util.find_spec (avoids slow full torch import)."""
    try:
        res = subprocess.run(
            [
                *python_cmd,
                "-c",
                "import importlib.util, sys; req = ('uvicorn', 'fastapi', 'torch', 'ultralytics'); "
                "sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in req) else 1)",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
        return res.returncode == 0
    except Exception:
        return False


def _filtered_requirements() -> list[str]:
    """Read requirements.txt excluding torch/torchvision/torchaudio and index flags (installed explicitly)."""
    pkgs: list[str] = []
    if not REQUIREMENTS_FILE.exists():
        return pkgs
    for raw in REQUIREMENTS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        low = line.lower()
        if low.startswith(("torch==", "torchvision==", "torchaudio==", "torch>", "torch<")):
            continue
        pkgs.append(line)
    return pkgs


def _has_pip(python_cmd: list[str]) -> bool:
    try:
        res = subprocess.run(
            [*python_cmd, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return res.returncode == 0
    except Exception:
        return False


def run_setup(
    os_type: str,
    use_system: bool,
    force_cpu: bool,
    force_cu118: bool,
    env: dict[str, str],
) -> list[str]:
    """Create/reuse OS venv and install PyTorch + project requirements."""
    base_py = find_base_python(os_type)
    base_ver = _probe_python_version(base_py)
    ver_str = f"{base_ver[0]}.{base_ver[1]}" if base_ver else "3"
    print(f"[1/4] He dieu hanh: {platform.system()} ({os_type}) | Python goc: {' '.join(base_py)} ({ver_str})")

    if use_system:
        target_py = base_py
        print(f"[2/4] Cai dat truc tiep vao Python he thong: {' '.join(target_py)}")
    else:
        venv_dir = resolve_venv_dir(os_type)
        vpy = venv_python_path(venv_dir, os_type)
        needs_create = not vpy.exists()
        if not needs_create and not _has_pip([str(vpy)]):
            subprocess.run([str(vpy), "-m", "ensurepip", "--upgrade"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not _has_pip([str(vpy)]):
                print(f"[WARN] Moi truong ao {venv_dir} bi loi/thieu pip — tao lai...")
                shutil.rmtree(venv_dir, ignore_errors=True)
                needs_create = True

        if needs_create:
            uv_bin = shutil.which("uv")
            created = False
            if uv_bin:
                print(f"[2/4] Tao moi truong ao bang uv (Python 3.11.6): {venv_dir}")
                uv_res = subprocess.run([uv_bin, "venv", "--python", "3.11.6", "--seed", str(venv_dir)])
                if uv_res.returncode == 0 and vpy.exists():
                    created = True
                else:
                    shutil.rmtree(venv_dir, ignore_errors=True)

            if not created:
                print(f"[2/4] Tao moi truong ao cho {os_type}: {venv_dir}")
                res = subprocess.run([*base_py, "-m", "venv", str(venv_dir)])
                if res.returncode != 0 or not _has_pip([str(vpy)]):
                    shutil.rmtree(venv_dir, ignore_errors=True)
                    if os_type == "linux":
                        pkg = f"python{ver_str}-venv" if base_ver else "python3-venv"
                        print(f"[HINT] Tren Ubuntu/Debian, cai goi venv cho Python {ver_str} bang:")
                        print(f"       sudo apt update && sudo apt install -y {pkg} python3-venv")
                        print("[HINT] Hoac dung uv de tu tai Python 3.11.6 (khong can apt/sudo):")
                        print("       curl -LsSf https://astral.sh/uv/install.sh | sh && ./start_app.sh --setup")
                    raise SystemExit(res.returncode or 1)
        else:
            print(f"[2/4] Su dung moi truong ao san co: {venv_dir}")
        target_py = [str(vpy)]

    gpu_name = detect_nvidia_gpu()
    install_cu118 = (force_cu118 or (gpu_name is not None and os_type in {"windows", "linux"})) and not force_cpu

    print(f"[3/4] Cai dat PyTorch ({'CUDA 11.8' if install_cu118 else 'CPU'}) & thu vien du an...")
    subprocess.run([*target_py, "-m", "pip", "install", "--upgrade", "pip"], env=env, check=False)

    if install_cu118:
        torch_cmd = [*target_py, "-m", "pip", "install", "--upgrade", "--force-reinstall", *PYTORCH_CU118_PKGS, "--index-url", PYTORCH_CU118_INDEX]
    elif os_type == "macos":
        torch_cmd = [*target_py, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio"]
    else:
        torch_cmd = [*target_py, "-m", "pip", "install", "--upgrade", *PYTORCH_CU118_PKGS, "--index-url", PYTORCH_CPU_INDEX]

    res = subprocess.run(torch_cmd, env=env)
    if res.returncode != 0:
        print("[ERROR] Cai dat PyTorch that bai!")
        raise SystemExit(res.returncode)

    other_pkgs = _filtered_requirements()
    if other_pkgs:
        res2 = subprocess.run([*target_py, "-m", "pip", "install", *other_pkgs], env=env)
        if res2.returncode != 0:
            print("[ERROR] Cai dat cac goi trong requirements.txt that bai!")
            raise SystemExit(res2.returncode)

    print("[4/4] Kiem tra moi truong...")
    subprocess.run(
        [
            *target_py,
            "-c",
            (
                "import platform, torch, torchvision, ultralytics; "
                "print('OS:', platform.system(), '| Python:', platform.python_version()); "
                "print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU Ready:', torch.cuda.is_available()); "
                "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); "
                "print('Ultralytics:', ultralytics.__version__)"
            ),
        ],
        env=env,
        check=False,
    )
    return target_py


def select_ready_python(os_type: str, env: dict[str, str], use_system: bool) -> list[str] | None:
    candidates: list[list[str]] = []
    if not use_system:
        venv_dir = resolve_venv_dir(os_type)
        vpy = venv_python_path(venv_dir, os_type)
        if vpy.exists():
            candidates.append([str(vpy)])
    base_py = find_base_python(os_type)
    if base_py not in candidates:
        candidates.append(base_py)
    if [sys.executable] not in candidates:
        candidates.append([sys.executable])

    for cmd in candidates:
        if has_core_deps(cmd, env):
            return cmd
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DERICH YOLO Studio — Unified Auto-OS Launcher (Windows / Linux / macOS)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1, use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--setup", action="store_true", help="Install/reinstall environment before launching")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment and exit")
    parser.add_argument("--system", action="store_true", help="Use system Python instead of virtualenv")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only PyTorch during --setup")
    parser.add_argument("--cu118", action="store_true", help="Force CUDA 11.8 PyTorch during --setup")
    parser.add_argument("--no-auto-setup", action="store_true", help="Do not auto-install when dependencies are missing")
    parser.add_argument("--check", action="store_true", help="Print detected OS/Python/CUDA info and exit")
    args = parser.parse_args()

    os_type = detect_os()
    env = build_runtime_env(os_type)
    gpu_name = detect_nvidia_gpu()
    cuda_root = detect_cuda_root(os_type)
    venv_dir = resolve_venv_dir(os_type)
    ready_py = select_ready_python(os_type, env, args.system)

    print("=" * 66)
    print("  DERICH YOLO Auto-Training Studio — Unified Auto-OS Launcher")
    print(f"  He dieu hanh : {platform.system()} {platform.release()} [{os_type}]")
    print(f"  Virtualenv   : {venv_dir} ({'san sang' if venv_python_path(venv_dir, os_type).exists() else 'chua tao'})")
    print(f"  GPU / CUDA   : {gpu_name or 'CPU'} | Toolkit: {cuda_root or 'khong co'}")
    print("=" * 66)

    if args.check:
        print(f"[CHECK] Base Python : {' '.join(find_base_python(os_type))}")
        print(f"[CHECK] Ready Python: {' '.join(ready_py) if ready_py else 'Chua cai du thu vien (chay --setup)'}")
        if os_type == "linux":
            print(f"[CHECK] LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', '(empty)')}")
        return

    target_py: list[str]
    if args.setup or args.setup_only:
        target_py = run_setup(os_type, args.system, args.cpu, args.cu118, env)
        if args.setup_only:
            print("[DONE] Cai dat hoan tat! Chay 'python run.py' (Windows) hoac 'python3 run.py' / './start_app.sh' (Linux).")
            return
    elif ready_py is not None:
        target_py = ready_py
    elif not args.no_auto_setup:
        print("[INFO] Chua thay du thu vien (uvicorn/fastapi/torch/ultralytics) — tu dong cai dat moi truong...")
        target_py = run_setup(os_type, args.system, args.cpu, args.cu118, env)
    else:
        print("[ERROR] Thieu thu vien. Hay chay: python3 run.py --setup")
        raise SystemExit(1)

    launch_cmd = [*target_py, str(MAIN_SCRIPT), "--host", args.host, "--port", str(args.port)]
    headless_linux = os_type == "linux" and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    if args.no_browser or headless_linux:
        launch_cmd.append("--no-browser")
        if headless_linux and not args.no_browser:
            print(f"[INFO] Phat hien Linux khong co GUI (headless) — mo trinh duyet tai http://{args.host}:{args.port}")
    if args.reload:
        launch_cmd.append("--reload")

    print(f"[RUN] {' '.join(launch_cmd)}")
    try:
        raise SystemExit(subprocess.call(launch_cmd, cwd=str(BASE_DIR), env=env))
    except KeyboardInterrupt:
        print("\n[INFO] Da dung DERICH YOLO Studio.")


if __name__ == "__main__":
    main()
