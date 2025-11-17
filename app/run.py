"""
Development hot reload script for PyQt6 application using watchdog.

Responsibilities:
- Launch `app.py` as a child process.
- Watch for changes in *.py and *.qss files inside relevant directories.
- Restart the child process when changes are detected (debounced to avoid rapid restarts).

Basic usage:
    python run.py

Options:
    --paths DIR1 DIR2 ...   Additional directories to watch.
    --ignore DIR1 DIR2 ...  Directories to ignore.
    --no-style              Do not watch .qss files.
    --delay 0.5             Debounce seconds (default 0.5).
    --once                  Start once and exit (syntax / startup check).

Interrupt with Ctrl+C.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

PY_PATTERNS = ["*.py"]
QSS_PATTERNS = ["*.qss"]


class DebouncedRestarter:
    """Manages debounced restarts."""

    def __init__(self, delay: float, restart_callback):
        self.delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._callback = restart_callback

    def trigger(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, self._callback)
            self._timer.daemon = True
            self._timer.start()


class HotReloadEventHandler(PatternMatchingEventHandler):
    def __init__(self, restarter: DebouncedRestarter, patterns: List[str]):
        super().__init__(patterns=patterns, ignore_directories=True)
        self.restarter = restarter

    def _log_event(self, event: FileSystemEvent, kind: str):
        rel = os.path.relpath(event.src_path, Path.cwd())
        print(f"[watch] {kind}: {rel}")

    def on_modified(self, event: FileSystemEvent):
        self._log_event(event, "MOD")
        self.restarter.trigger()

    def on_created(self, event: FileSystemEvent):
        self._log_event(event, "NEW")
        self.restarter.trigger()

    def on_deleted(self, event: FileSystemEvent):
        self._log_event(event, "DEL")
        self.restarter.trigger()


class AppProcess:
    def __init__(self):
        self._proc: subprocess.Popen | None = None

    def start(self):
        self.stop()
        print("[run] Starting app...")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=str(Path(__file__).parent),
            env=env,
        )
        print(f"[run] Child PID: {self._proc.pid}")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            print("[run] Stopping child process...")
            try:
                self._proc.terminate()
                for _ in range(10):
                    if self._proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if self._proc.poll() is None:
                    print("[run] Force kill.")
                    self._proc.kill()
            except Exception as e:
                print(f"[run] Error stopping process: {e}")
        self._proc = None

    def restart(self):
        print("[run] Restart requested.")
        self.start()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PyQt6 hot reload with watchdog")
    p.add_argument(
        "--paths", nargs="*", default=[], help="Additional directories to watch"
    )
    p.add_argument("--ignore", nargs="*", default=[], help="Directories to ignore")
    p.add_argument("--no-style", action="store_true", help="Do not watch .qss files")
    p.add_argument("--delay", type=float, default=0.5, help="Debounce seconds")
    p.add_argument(
        "--once", action="store_true", help="Start once and exit (startup test)"
    )
    return p


def collect_watch_paths(extra: List[str], ignore: List[str]) -> List[Path]:
    base = Path(__file__).parent
    default_dirs = ["core", "gui", "models", "styles", "utils"]
    paths = []
    paths.append(base)
    for d in default_dirs + extra:
        p = base / d
        if p.exists() and p.is_dir() and d not in ignore:
            paths.append(p)
    return paths


def main():
    args = build_arg_parser().parse_args()
    app_proc = AppProcess()

    if args.once:
        app_proc.start()
        time.sleep(1)
        app_proc.stop()
        print("[run] --once execution completed.")
        return

    patterns = PY_PATTERNS.copy()
    if not args.no_style:
        patterns += QSS_PATTERNS

    restarter = DebouncedRestarter(args.delay, app_proc.restart)
    handler = HotReloadEventHandler(restarter, patterns=patterns)

    watch_paths = collect_watch_paths(args.paths, args.ignore)
    if not watch_paths:
        print("[warn] No directories to watch.")

    observer = Observer()
    for p in watch_paths:
        observer.schedule(handler, str(p), recursive=True)
        print(f"[watch] Watching: {p}")

    def handle_sigint(sig, frame):  # noqa: ARG001
        print("\n[run] Exit signal received. Shutting down...")
        observer.stop()
        app_proc.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    app_proc.start()
    observer.start()

    print("[run] Hot reload active. Press Ctrl+C to exit.")
    try:
        while observer.is_alive():
            observer.join(timeout=0.5)
            # if the process dies (uncaught exception), keep it dead until a file change triggers restart.
            if not app_proc.alive():
                pass
    finally:
        observer.stop()
        observer.join()
        app_proc.stop()
        print("[run] Finished.")


if __name__ == "__main__":
    main()
