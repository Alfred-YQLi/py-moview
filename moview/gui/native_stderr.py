from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections.abc import Iterator


MACOS_TSM_WARNING = "TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED"
MACOS_IMK_WAKE_WARNING = "error messaging the mach port for IMKCFRunLoopWakeUpReliable"
QT_KEYMAPPER_PREFIX = "qt.qpa.keymapper: Mismatch between Cocoa "


def _should_suppress_native_stderr(line: str) -> bool:
    if MACOS_TSM_WARNING in line or MACOS_IMK_WAKE_WARNING in line:
        return True
    start = line.find(QT_KEYMAPPER_PREFIX)
    if start < 0:
        return False
    message = line[start:]
    return (
        " and Carbon " in message
        and " for virtual key " in message
        and "QFlags<Qt::KeyboardModifier>(" in message
    )


@contextlib.contextmanager
def filter_macos_gui_warnings(enabled: bool = True) -> Iterator[None]:
    """Suppress known harmless macOS input-method diagnostics.

    These messages can be written directly to file descriptor 2, bypassing
    Python logging. Every line outside the narrow allowlist is forwarded.
    """
    if not enabled or sys.platform != "darwin":
        yield
        return

    original_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    stop_event = threading.Event()

    def forward_stderr() -> None:
        pending = b""
        with os.fdopen(read_fd, "rb", closefd=True) as reader:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    raw_line, pending = pending.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace")
                    if not _should_suppress_native_stderr(line):
                        os.write(original_stderr_fd, raw_line + b"\n")
            if pending:
                line = pending.decode("utf-8", errors="replace")
                if not _should_suppress_native_stderr(line):
                    os.write(original_stderr_fd, pending)
        stop_event.set()

    thread = threading.Thread(target=forward_stderr, name="moview-stderr-filter", daemon=True)
    thread.start()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        try:
            sys.stderr.flush()
        except Exception:
            pass
        stop_event.wait(timeout=0.25)
        os.close(original_stderr_fd)
