"""Platform helpers: macOS stderr filtering and the Linux xcb preflight."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from moview.gui.platform.linux_qt import (
    _missing_shared_libraries,
    _parse_missing_libraries,
    linux_qt_platform_issue,
)
from moview.gui.platform.macos_stderr import _should_suppress_native_stderr, filter_macos_gui_warnings


class MacOSStderrTests(unittest.TestCase):
    def test_macos_gui_warning_filter_predicate(self) -> None:
        suppressed = (
            "TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED(-1)",
            "2026-07-10 python[1:2] error messaging the mach port "
            "for IMKCFRunLoopWakeUpReliable",
            "qt.qpa.keymapper: Mismatch between Cocoa 'd' and Carbon '\\x0' "
            "for virtual key 2 with QFlags<Qt::KeyboardModifier>(NoModifier)",
        )
        for line in suppressed:
            with self.subTest(line=line):
                self.assertTrue(_should_suppress_native_stderr(line))
        preserved = (
            "Traceback (most recent call last):",
            "qt.qpa.keymapper: keyboard initialization failed",
            "Mismatch between Cocoa and Carbon without the Qt category",
            "OpenGL shader compilation failed",
        )
        for line in preserved:
            with self.subTest(line=line):
                self.assertFalse(_should_suppress_native_stderr(line))

    def test_macos_gui_warning_filter_preserves_other_stderr(self) -> None:
        capture_read, capture_write = os.pipe()
        saved_stderr = os.dup(2)
        restored = False
        try:
            os.dup2(capture_write, 2)
            os.close(capture_write)
            with patch.object(sys, "platform", "darwin"):
                with filter_macos_gui_warnings():
                    os.write(2, b"TSMSendMessageToUIServer: CFMessagePortSendRequest FAILED(-1)\n")
                    os.write(2, b"error messaging the mach port for IMKCFRunLoopWakeUpReliable\n")
                    os.write(
                        2,
                        b"qt.qpa.keymapper: Mismatch between Cocoa 'd' and Carbon '\\x0' "
                        b"for virtual key 2 with QFlags<Qt::KeyboardModifier>(NoModifier)\n",
                    )
                    os.write(2, b"keep this stderr line\n")
                    os.write(2, b"Traceback (most recent call last):\n")
            os.dup2(saved_stderr, 2)
            restored = True
            with os.fdopen(capture_read, "rb") as reader:
                output = reader.read().decode("utf-8", errors="replace")
        finally:
            if not restored:
                os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        self.assertNotIn("TSMSendMessageToUIServer", output)
        self.assertNotIn("IMKCFRunLoopWakeUpReliable", output)
        self.assertNotIn("qt.qpa.keymapper", output)
        self.assertIn("keep this stderr line", output)
        self.assertIn("Traceback (most recent call last):", output)


class LinuxQtPreflightTests(unittest.TestCase):
    def test_missing_ldd_libraries_are_parsed_once(self) -> None:
        output = """
            libxcb-cursor.so.0 => not found
            libxcb-image.so.0 => /lib64/libxcb-image.so.0 (0x0001)
            libxcb-cursor.so.0 => not found
            libxkbcommon-x11.so.0 => not found
        """
        self.assertEqual(
            _parse_missing_libraries(output),
            ("libxcb-cursor.so.0", "libxkbcommon-x11.so.0"),
        )

    def test_ldd_runs_with_stable_locale(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ldd", "libqxcb.so"],
            returncode=0,
            stdout="libxcb-cursor.so.0 => not found\n",
            stderr="",
        )
        with (
            patch("moview.gui.platform.linux_qt.shutil.which", return_value="/usr/bin/ldd"),
            patch("moview.gui.platform.linux_qt.subprocess.run", return_value=completed) as run,
        ):
            missing = _missing_shared_libraries(
                Path("/qt/plugins/libqxcb.so"),
                {"LD_LIBRARY_PATH": "/custom/lib"},
            )

        self.assertEqual(missing, ("libxcb-cursor.so.0",))
        self.assertEqual(run.call_args.args[0], ["/usr/bin/ldd", "/qt/plugins/libqxcb.so"])
        self.assertEqual(run.call_args.kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(run.call_args.kwargs["env"]["LD_LIBRARY_PATH"], "/custom/lib")
        self.assertFalse(run.call_args.kwargs["check"])

    def test_non_linux_and_non_xcb_platforms_skip_preflight(self) -> None:
        with patch("moview.gui.platform.linux_qt._find_xcb_plugin") as find_plugin:
            self.assertIsNone(linux_qt_platform_issue(environ={}, host_platform="darwin"))
            self.assertIsNone(
                linux_qt_platform_issue(
                    environ={"QT_QPA_PLATFORM": "offscreen"},
                    host_platform="linux",
                )
            )
        find_plugin.assert_not_called()

    def test_complete_xcb_dependencies_allow_startup(self) -> None:
        with (
            patch(
                "moview.gui.platform.linux_qt._find_xcb_plugin",
                return_value=Path("/site-packages/PyQt6/Qt6/plugins/platforms/libqxcb.so"),
            ),
            patch("moview.gui.platform.linux_qt._missing_shared_libraries", return_value=()),
        ):
            issue = linux_qt_platform_issue(environ={}, host_platform="linux")

        self.assertIsNone(issue)

    def test_centos_8_issue_names_native_package(self) -> None:
        with (
            patch(
                "moview.gui.platform.linux_qt._find_xcb_plugin",
                return_value=Path("/site-packages/PyQt6/Qt6/plugins/platforms/libqxcb.so"),
            ),
            patch(
                "moview.gui.platform.linux_qt._missing_shared_libraries",
                return_value=("libxcb-cursor.so.0",),
            ),
            patch(
                "moview.gui.platform.linux_qt._os_release",
                return_value={"ID": "centos", "ID_LIKE": "rhel fedora", "VERSION_ID": "8"},
            ),
        ):
            issue = linux_qt_platform_issue(environ={}, host_platform="linux")

        self.assertIsNotNone(issue)
        self.assertIn("libxcb-cursor.so.0", issue)
        self.assertIn("sudo dnf install epel-release", issue)
        self.assertIn("sudo dnf install xcb-util-cursor", issue)
        self.assertIn("cannot be installed by pip", issue)


if __name__ == "__main__":
    unittest.main()
