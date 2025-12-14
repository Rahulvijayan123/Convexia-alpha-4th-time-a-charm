from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque


logger = logging.getLogger(__name__)

# Context propagated across asyncio tasks; used to route log records to the right run stream.
UI_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("ui_run_id", default=None)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunState:
    ui_run_id: str
    created_at_iso: str = field(default_factory=_utc_iso_now)
    status: str = "created"  # created|running|completed|failed
    discovery_run_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    _seq: int = 0
    _t0: float = field(default_factory=time.perf_counter)
    _t_last: float = field(default_factory=time.perf_counter)
    _queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    _buffer: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=6000))

    def _event_base(self) -> dict[str, Any]:
        now = time.perf_counter()
        elapsed_ms = int((now - self._t0) * 1000)
        delta_ms = int((now - self._t_last) * 1000)
        self._t_last = now
        self._seq += 1
        return {
            "ui_run_id": self.ui_run_id,
            "seq": self._seq,
            "ts": _utc_iso_now(),
            "elapsed_ms": elapsed_ms,
            "delta_ms": delta_ms,
        }

    def publish(self, *, level: str, kind: str, message: str, data: dict[str, Any] | None = None) -> None:
        evt = self._event_base()
        evt.update(
            {
                "level": level,
                "kind": kind,
                "message": message,
                "data": data or {},
            }
        )
        self._buffer.append(evt)
        # Non-blocking best-effort (queue is unbounded, but keep it safe anyway)
        try:
            self._queue.put_nowait(evt)
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "ui_run_id": self.ui_run_id,
            "created_at": self.created_at_iso,
            "status": self.status,
            "discovery_run_id": self.discovery_run_id,
            "error": self.error,
        }

    def buffered_events(self) -> list[dict[str, Any]]:
        return list(self._buffer)

    async def next_event(self, *, timeout_s: float = 15.0) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None


class _RunLogHandler(logging.Handler):
    """
    Captures python logging records for a single UI run, routed via UI_RUN_ID contextvar.
    """

    def __init__(self, state: RunState):
        super().__init__(level=logging.DEBUG)
        self._state = state
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            current = UI_RUN_ID.get()
            if current != self._state.ui_run_id:
                return
            # Keep only our app logs (avoid uvicorn noise)
            if not (record.name or "").startswith("drug_asset_discovery"):
                return

            msg = record.getMessage()
            exc = None
            if record.exc_info:
                exc = "".join(traceback.format_exception(*record.exc_info)).strip()

            self._state.publish(
                level=record.levelname,
                kind="log",
                message=msg,
                data={
                    "logger": record.name,
                    "file": record.pathname,
                    "line": record.lineno,
                    "func": record.funcName,
                    "thread": record.threadName,
                    "exc": exc,
                },
            )
        except Exception:
            # Never crash the app due to logging
            return


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = asyncio.Lock()

    async def create_run(self) -> RunState:
        async with self._lock:
            ui_run_id = str(uuid.uuid4())
            st = RunState(ui_run_id=ui_run_id)
            self._runs[ui_run_id] = st
            st.publish(level="INFO", kind="ui", message="UI run created", data={"ui_run_id": ui_run_id})
            return st

    async def get(self, ui_run_id: str) -> RunState | None:
        async with self._lock:
            return self._runs.get(ui_run_id)

    async def start(
        self,
        *,
        ui_run_id: str,
        coro_factory,
    ) -> None:
        st = await self.get(ui_run_id)
        if not st:
            return
        if st.status in ("running", "completed", "failed"):
            return

        async def _runner() -> None:
            token = UI_RUN_ID.set(st.ui_run_id)
            # IMPORTANT: Do NOT change the root logger level.
            # Setting root to DEBUG causes third-party libs (httpx/httpcore/hpack) to log
            # request headers, which can include sensitive credentials (e.g. Supabase keys).
            app_logger = logging.getLogger("drug_asset_discovery")
            old_level = app_logger.level
            old_propagate = app_logger.propagate
            handler = _RunLogHandler(st)
            app_logger.addHandler(handler)
            app_logger.setLevel(logging.DEBUG)
            # Keep detailed logs in the UI only (avoid noisy/unsafe console logs).
            app_logger.propagate = False
            try:
                st.status = "running"
                st.publish(level="INFO", kind="run", message="Run started", data={"status": st.status})
                result = await coro_factory(st)
                st.result = result
                st.status = "completed"
                st.publish(
                    level="INFO",
                    kind="run",
                    message="Run completed",
                    data={"status": st.status, "result_keys": list(result.keys())},
                )
            except Exception as e:
                st.status = "failed"
                st.error = str(e)
                st.publish(
                    level="ERROR",
                    kind="run",
                    message="Run failed",
                    data={"status": st.status, "error": str(e), "traceback": traceback.format_exc()},
                )
            finally:
                try:
                    app_logger.removeHandler(handler)
                except Exception:
                    pass
                app_logger.setLevel(old_level)
                app_logger.propagate = old_propagate
                UI_RUN_ID.reset(token)

        asyncio.create_task(_runner())

    async def sse_stream(self, ui_run_id: str):
        st = await self.get(ui_run_id)
        if not st:
            return

        # Immediately replay buffered events (so refresh doesn't lose context)
        for evt in st.buffered_events():
            yield self._format_sse(evt)

        # Then stream new ones
        while True:
            evt = await st.next_event(timeout_s=15.0)
            if evt is None:
                # Keepalive
                yield "event: ping\ndata: {}\n\n"
                # If run is terminal and we haven't seen updates for a while, exit.
                if st.status in ("completed", "failed"):
                    # one last status event before closing
                    yield self._format_sse(
                        {
                            "ui_run_id": st.ui_run_id,
                            "seq": st._seq + 1,
                            "ts": _utc_iso_now(),
                            "elapsed_ms": int((time.perf_counter() - st._t0) * 1000),
                            "delta_ms": 0,
                            "level": "INFO",
                            "kind": "run",
                            "message": "Run terminal state reached; closing stream",
                            "data": {"status": st.status},
                        }
                    )
                    break
                continue
            yield self._format_sse(evt)

    def _format_sse(self, evt: dict[str, Any]) -> str:
        payload = json.dumps(evt, ensure_ascii=False)
        return f"event: message\ndata: {payload}\n\n"

