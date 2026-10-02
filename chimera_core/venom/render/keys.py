"""Single-key terminal input without echo (POSIX), shared by watch and the discovery animation."""

from __future__ import annotations

from typing import Optional


class Keys:
    """Single-key input without echo (POSIX terminals). Arrow keys arrive as 'up' / 'down'."""

    def __init__(self) -> None:
        self.fd = None
        self.saved = None

    def __enter__(self):
        import sys
        import termios
        import tty

        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        import termios

        if self.fd is not None and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self, timeout: float) -> Optional[str]:
        import os
        import select

        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        ch = os.read(self.fd, 1).decode("utf-8", errors="ignore")
        if ch == "\x1b":
            more, _, _ = select.select([self.fd], [], [], 0.03)
            if not more:
                return "esc"
            seq = os.read(self.fd, 2).decode("utf-8", errors="ignore")
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\t":
            return "tab"
        if ch in ("\x7f", "\x08"):
            return "backspace"
        if ch == "\x03":
            return "ctrl-c"
        return ch
