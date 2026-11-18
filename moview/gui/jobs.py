"""Run callables on worker threads and deliver results on the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from .qt import QtCore, Signal


class JobRunner(QtCore.QObject):
    """Thin wrapper around a thread pool.

    ``on_done(result)`` / ``on_error(exception)`` run on the thread that owns
    the runner (the GUI thread) via a queued signal, so they may touch widgets.
    Cancelled futures report nothing.
    """

    _finished = Signal(object)

    def __init__(self, max_workers: int, name: str, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix=name)
        self._pending: set[Future] = set()
        self._closed = False
        self._finished.connect(self._deliver, QtCore.Qt.ConnectionType.QueuedConnection)

    def submit(self, work: Callable[[], object], on_done: Callable[[object], None], on_error: Callable[[BaseException], None] | None = None) -> Future | None:
        if self._closed:
            return None
        future = self._executor.submit(work)
        self._pending.add(future)
        future.add_done_callback(lambda f: self._finished.emit((f, on_done, on_error)))
        return future

    @property
    def busy(self) -> bool:
        return any(not future.done() for future in self._pending)

    def cancel_pending(self) -> None:
        """Drop queued (not yet running) jobs."""
        for future in list(self._pending):
            future.cancel()

    def _deliver(self, payload) -> None:
        future, on_done, on_error = payload
        self._pending.discard(future)
        if future.cancelled():
            return
        error = future.exception()
        if error is None:
            on_done(future.result())
        elif on_error is not None:
            on_error(error)

    def shutdown(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
