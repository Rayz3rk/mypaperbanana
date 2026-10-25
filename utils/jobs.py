"""Per-session background work; Streamlit remains responsive during generation."""

import asyncio
import threading
from datetime import datetime
from uuid import uuid4

from utils.results import RESULTS_DIR, write_history


class BatchJob:
    def __init__(self, inputs: list[dict], settings: dict, retained_results: list[dict] | None = None):
        self.id = uuid4().hex
        self.inputs = inputs
        self.settings = settings  # Public settings only; credentials never enter history.
        self.created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.path = RESULTS_DIR / f"demo_{datetime.now():%Y%m%d_%H%M%S}_{self.id[:8]}.json"
        self.results: list[dict] = []
        self.retained_results = list(retained_results or [])
        self.stages: dict = {}
        self.usage: dict = {}
        self.status = "queued"
        self.error = ""
        self.cancel_event = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return {"version": 2, "id": self.id, "created_at": self.created_at, "status": self.status,
                    "settings": dict(self.settings), "inputs": list(self.inputs),
                    "results": self.retained_results + list(self.results), "stages": dict(self.stages), "usage": dict(self.usage),
                    "error": self.error, "total": len(self.inputs) + len(self.retained_results)}

    def checkpoint(self) -> None:
        write_history(self.path, self.snapshot())

    def report_stage(self, candidate_id, stage: str) -> None:
        with self._lock:
            self.stages[str(candidate_id)] = stage

    def report_usage(self, usage: dict) -> None:
        with self._lock:
            self.usage = dict(usage)

    def add_result(self, result: dict) -> None:
        with self._lock:
            self.results.append(result)
        self.checkpoint()

    def start(self, runner) -> "BatchJob":
        if self._thread is not None:
            raise RuntimeError("任务已经启动。")

        async def run():
            with self._lock:
                self.status = "running"
            self.checkpoint()
            work = asyncio.create_task(runner(self))

            async def watch_cancel():
                while not work.done():
                    if self.cancel_event.is_set():
                        work.cancel()
                        return
                    await asyncio.sleep(0.15)

            watcher = asyncio.create_task(watch_cancel())
            try:
                await work
                with self._lock:
                    self.status = "partial" if any(r.get("status") == "failed" for r in self.results) else "completed"
            except asyncio.CancelledError:
                with self._lock:
                    self.status = "cancelled"
            except Exception as exc:
                # Do not persist raw gateway errors, which can echo request credentials.
                with self._lock:
                    self.status = "failed"
                    self.error = f"任务中断（{type(exc).__name__}），已保留完成的候选。"
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                self.checkpoint()

        self._thread = threading.Thread(target=lambda: asyncio.run(run()), name=f"paperbanana-{self.id[:8]}", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
