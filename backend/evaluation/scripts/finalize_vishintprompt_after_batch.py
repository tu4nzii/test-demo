from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
GRID_ROOT = Path(r"F:\program\grid")
LOG_DIR = BACKEND / "evaluation" / "tmp"
LOG_DIR.mkdir(parents=True, exist_ok=True)


GRID_FILES = [
    "grid_adjudication.py",
    "grid_bindings.py",
    "grid_geometry.py",
    "grid_io.py",
    "grid_line_filter.py",
    "grid_masks.py",
    "grid_math.py",
    "grid_mllm.py",
    "grid_ocr.py",
    "grid_visual.py",
]


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
    ]
    return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ensure_grid_modules(log) -> None:
    target_dir = BACKEND / "Grid_generation"
    for name in GRID_FILES:
        src = GRID_ROOT / name
        dst = target_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        if not dst.exists() or sha1(src) != sha1(dst):
            dst.write_bytes(src.read_bytes())
            print(f"[sync] {name}", file=log, flush=True)
        else:
            print(f"[same] {name}", file=log, flush=True)


def run_step(log, args: list[str]) -> None:
    print(f"[run] {' '.join(args)}", file=log, flush=True)
    completed = subprocess.run(
        args,
        cwd=ROOT,
        stdout=log,
        stderr=log,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed with code {completed.returncode}: {' '.join(args)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "finalize_vishintprompt_after_batch.log")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=BACKEND / "evaluation" / "results" / "vishintprompt_latest_metrics",
    )
    args = parser.parse_args()

    with args.log.open("a", encoding="utf-8") as log:
        print(f"\n[start] {time.strftime('%Y-%m-%d %H:%M:%S')}", file=log, flush=True)
        if args.wait_pid:
            print(f"[wait] pid={args.wait_pid}", file=log, flush=True)
            while process_exists(args.wait_pid):
                time.sleep(max(5, args.poll_seconds))
        print("[wait] done", file=log, flush=True)

        ensure_grid_modules(log)
        run_step(
            log,
            [
                sys.executable,
                "backend/evaluation/scripts/run_vishintprompt_full_grid_encryption.py",
            ],
        )
        run_step(
            log,
            [
                sys.executable,
                "backend/evaluation/scripts/evaluate_vishintprompt_latest_metrics.py",
                "--output",
                str(args.metrics_output),
            ],
        )
        print(f"[complete] {time.strftime('%Y-%m-%d %H:%M:%S')}", file=log, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
