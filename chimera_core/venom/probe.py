"""
Every interaction Venom has with the host goes through a HostProbe.

Discovery is read-only: the probe can stat, list and read files (bounded size) and run a
small allowlist of read-only commands. It never writes, never executes discovered code and
never opens a network connection. The probe records what it opened so tests can prove that
session transcripts and similar files were only counted, never read.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

MAX_READ_BYTES = 1_000_000

# Read-only commands Venom may run. Anything else is refused.
ALLOWED_COMMANDS = {
    "ps", "crontab", "systemctl", "journalctl", "launchctl", "docker", "lsof", "ss", "id",
}

DEFAULT_EXCLUDES = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", "build", "dist", "site-packages", ".idea", ".cache", ".next", "target",
    ".eggs", ".terraform", "Library", ".Trash", ".npm", ".cargo", ".rustup", ".pyenv", ".local",
    ".docker", "Pictures", "Music", "Movies", ".csl", "tests", "test", "testing",
}


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


class HostProbe:
    """Interface. Paths are absolute POSIX strings in the probe's own namespace."""

    mode = "host"  # host | folder | fixture

    def __init__(self) -> None:
        self.opened: List[str] = []
        self.commands_run: List[str] = []
        self.spawned: List[str] = []  # MCP servers started (only with --probe)
        self.http_calls: List[str] = []
        self.allow_spawn = False  # set only after the operator confirmed --probe
        self.not_readable = 0

    # identity --------------------------------------------------------------
    def hostname(self) -> str: raise NotImplementedError
    def os_name(self) -> str: raise NotImplementedError
    def home(self) -> str: raise NotImplementedError
    def user(self) -> str: raise NotImplementedError
    def now(self) -> datetime: raise NotImplementedError

    # filesystem --------------------------------------------------------------
    def exists(self, path: str) -> bool: raise NotImplementedError
    def is_dir(self, path: str) -> bool: raise NotImplementedError
    def stat(self, path: str) -> Optional[Tuple[int, float]]: raise NotImplementedError
    def list_dir(self, path: str) -> List[str]: raise NotImplementedError
    def _read(self, path: str, limit: int) -> Optional[str]: raise NotImplementedError

    def read_text(self, path: str, limit: int = MAX_READ_BYTES) -> Optional[str]:
        st = self.stat(path)
        if st is None or st[0] > limit:
            return None
        text = self._read(path, limit)
        if text is None:
            self.not_readable += 1
            return None
        self.opened.append(path)
        return text

    def walk(
        self, root: str, excludes: Sequence[str] = tuple(DEFAULT_EXCLUDES), suffixes: Sequence[str] = (".py",),
        max_files: int = 20000,
    ) -> Iterator[str]:
        """Yield files under root with one of the suffixes, skipping excluded folders."""
        stack = [root]
        count = 0
        excl = set(excludes)
        while stack:
            d = stack.pop()
            try:
                names = sorted(self.list_dir(d), reverse=True)
            except PermissionError:
                self.not_readable += 1
                continue
            for name in names:
                p = _join(d, name)
                if self.is_dir(p):
                    if name in excl or name.startswith("."):
                        continue
                    stack.append(p)
                elif name.endswith(tuple(suffixes)):
                    count += 1
                    if count > max_files:
                        return
                    yield p

    # commands --------------------------------------------------------------
    def run(self, argv: List[str], timeout: float = 10.0) -> Optional[CommandResult]:
        """Run an allowlisted read-only command. None when the command is unavailable."""
        if not argv or argv[0] not in ALLOWED_COMMANDS:
            raise PermissionError(f"venom probe refuses to run {argv[:1]}")
        self.commands_run.append(" ".join(argv))
        return self._run(argv, timeout)

    def _run(self, argv: List[str], timeout: float) -> Optional[CommandResult]: raise NotImplementedError

    def process_table(self) -> Optional[List[Tuple[int, Optional[int], str, str, str]]]:
        """(pid, uid, user, elapsed, args) rows from psutil when installed; None otherwise."""
        return None


    # live MCP (L7, only with --probe) ----------------------------------------------
    def mcp_stdio(self, argv: List[str], env: Dict[str, str], messages: List[dict], timeout: float = 10.0) -> Optional[List[dict]]:
        """Start one stdio MCP server, exchange JSON-RPC messages, stop it. Needs allow_spawn."""
        if not self.allow_spawn:
            raise PermissionError("starting MCP servers requires --probe and confirmation")
        self.spawned.append(" ".join(argv))
        return self._mcp_stdio(argv, env, messages, timeout)

    def _mcp_stdio(self, argv, env, messages, timeout):
        import threading

        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    env={**os.environ, **env}, text=True, bufsize=1)
        except (OSError, ValueError):
            return None
        out: List[dict] = []
        want = {m["id"] for m in messages if "id" in m}

        def reader():
            for line in proc.stdout:  # type: ignore[union-attr]
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("id") in want:
                    out.append(msg)
                    if len(out) == len(want):
                        return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for i, m in enumerate(messages):
                proc.stdin.write(json.dumps(m) + "\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
                if "id" in m:
                    deadline = timeout
                    while deadline > 0 and not any(r.get("id") == m["id"] for r in out):
                        t.join(0.05)
                        deadline -= 0.05
        except (OSError, ValueError):
            pass
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        return out

    def mcp_http(self, url: str, messages: List[dict], timeout: float = 5.0) -> Optional[List[dict]]:
        """JSON-RPC over MCP streamable HTTP, loopback addresses only, only with --probe."""
        from urllib.parse import urlparse

        if not self.allow_spawn:
            raise PermissionError("querying MCP servers requires --probe and confirmation")
        host = (urlparse(url).hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1") and not host.startswith("127."):
            return None
        self.http_calls.append(url)
        return self._mcp_http(url, messages, timeout)

    def _mcp_http(self, url, messages, timeout):
        import urllib.error
        import urllib.request

        out: List[dict] = []
        session: Optional[str] = None
        for m in messages:
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            if session:
                headers["Mcp-Session-Id"] = session
            req = urllib.request.Request(url, data=json.dumps(m).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    session = r.headers.get("Mcp-Session-Id") or session
                    body = r.read(2_000_000).decode("utf-8", errors="replace")
            except (urllib.error.URLError, OSError, ValueError):
                return out or None
            for chunk in [body] + [l[5:].strip() for l in body.splitlines() if l.startswith("data:")]:
                try:
                    msg = json.loads(chunk)
                except ValueError:
                    continue
                if isinstance(msg, dict) and "id" in msg:
                    out.append(msg)
                    break
        return out


def _join(d: str, name: str) -> str:
    return str(PurePosixPath(d) / name)


class LocalHostProbe(HostProbe):
    def __init__(self, mode: str = "host") -> None:
        super().__init__()
        self.mode = mode

    def hostname(self) -> str:
        return socket.gethostname().split(".")[0]

    def os_name(self) -> str:
        return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system(), platform.system().lower())

    def home(self) -> str:
        return str(Path.home())

    def user(self) -> str:
        try:
            import getpass
            return getpass.getuser()
        except Exception:
            return "unknown"

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path) and not os.path.islink(path)

    def stat(self, path: str) -> Optional[Tuple[int, float]]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return st.st_size, st.st_mtime

    def list_dir(self, path: str) -> List[str]:
        try:
            return os.listdir(path)
        except FileNotFoundError:
            return []
        except (PermissionError, OSError) as e:
            raise PermissionError(str(e))

    def _read(self, path: str, limit: int) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                data = f.read(limit)
        except OSError:
            return None
        if b"\x00" in data[:4096]:
            return None
        return data.decode("utf-8", errors="replace")

    def _run(self, argv: List[str], timeout: float) -> Optional[CommandResult]:
        try:
            r = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                env={**os.environ, "LC_ALL": "C", "SYSTEMD_PAGER": "", "PAGER": "cat"},
            )
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
            return None
        return CommandResult(r.returncode, r.stdout, r.stderr)

    def process_table(self):
        try:
            import psutil  # type: ignore  # optional: pip install csl-core[venom]
        except ImportError:
            return None
        rows = []
        for p in psutil.process_iter(["pid", "uids", "username", "cmdline"]):
            try:
                info = p.info
                cmd = " ".join(info.get("cmdline") or [])
                if cmd:
                    uid = info["uids"].real if info.get("uids") else None
                    rows.append((info["pid"], uid, info.get("username") or "?", "", cmd))
            except Exception:
                continue
        self.commands_run.append("psutil.process_iter")
        return rows


class FixtureHostProbe(HostProbe):
    """
    A virtual host backed by a folder:

        <fixture>/host.json      {"hostname", "os", "home", "user", "now"}
        <fixture>/commands.json  {"ps -eo ...": "<stdout>", ...}  (absent key = unavailable)
        <fixture>/fs/...         the host filesystem, "/" maps to fs/
    """

    mode = "fixture"

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        meta_path = self.root / "host.json"
        self.meta: Dict[str, str] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        cmd_path = self.root / "commands.json"
        self.commands: Dict[str, object] = json.loads(cmd_path.read_text(encoding="utf-8")) if cmd_path.exists() else {}

    @staticmethod
    def is_fixture(path: str) -> bool:
        p = Path(path)
        return (p / "host.json").is_file() and (p / "fs").is_dir()

    def _real(self, path: str) -> Path:
        return self.root / "fs" / path.lstrip("/")

    def hostname(self) -> str:
        return self.meta.get("hostname", "fixture-host")

    def os_name(self) -> str:
        return self.meta.get("os", "linux")

    def home(self) -> str:
        return self.meta.get("home", "/home/operator")

    def user(self) -> str:
        return self.meta.get("user", "operator")

    def now(self) -> datetime:
        raw = self.meta.get("now", "2026-10-01T12:00:00+00:00")
        return datetime.fromisoformat(raw)

    def exists(self, path: str) -> bool:
        return self._real(path).exists()

    def is_dir(self, path: str) -> bool:
        return self._real(path).is_dir()

    def stat(self, path: str) -> Optional[Tuple[int, float]]:
        p = self._real(path)
        if not p.exists():
            return None
        mtimes = self.meta.get("mtimes", {})
        mtime = mtimes.get(path) if isinstance(mtimes, dict) else None
        if mtime is None:
            mtime = self.now().timestamp() - 86400
        elif isinstance(mtime, str):
            mtime = datetime.fromisoformat(mtime).timestamp()
        return p.stat().st_size, float(mtime)

    def list_dir(self, path: str) -> List[str]:
        p = self._real(path)
        return sorted(os.listdir(p)) if p.is_dir() else []

    def _read(self, path: str, limit: int) -> Optional[str]:
        try:
            return self._real(path).read_bytes()[:limit].decode("utf-8", errors="replace")
        except OSError:
            return None

    def _mcp_stdio(self, argv, env, messages, timeout):
        canned = (self.meta.get("mcp") or {}).get(" ".join(argv)) if isinstance(self.meta.get("mcp"), dict) else None
        return _canned(canned, messages)

    def _mcp_http(self, url, messages, timeout):
        canned = (self.meta.get("mcp") or {}).get(url) if isinstance(self.meta.get("mcp"), dict) else None
        return _canned(canned, messages)

    def _run(self, argv: List[str], timeout: float) -> Optional[CommandResult]:
        key = " ".join(argv)
        val = self.commands.get(key)
        if val is None:
            # prefix match lets one canned output serve e.g. journalctl with any --since
            for k, v in self.commands.items():
                if key.startswith(k):
                    val = v
                    break
        if val is None:
            return None
        if isinstance(val, dict):
            return CommandResult(int(val.get("returncode", 0)), str(val.get("stdout", "")), str(val.get("stderr", "")))
        return CommandResult(0, str(val))


def _canned(tools, messages):
    if tools is None:
        return None
    out = []
    for m in messages:
        if m.get("method") == "initialize":
            out.append({"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": "2025-06-18", "capabilities": {}}})
        elif m.get("method") == "tools/list":
            out.append({"jsonrpc": "2.0", "id": m["id"], "result": {"tools": tools}})
    return out


def no_color_requested() -> bool:
    """https://no-color.org: any non-empty NO_COLOR disables colour."""
    return bool(os.environ.get("NO_COLOR"))


def animation_disabled() -> bool:
    """CI systems and CSL_NO_ANIM turn the discovery animation off."""
    return bool(os.environ.get("CSL_NO_ANIM") or os.environ.get("CI"))


def probe_for(root: Optional[str]) -> Tuple[HostProbe, List[str]]:
    """Pick the probe and scan roots for a --root argument (None = this host)."""
    if root:
        if FixtureHostProbe.is_fixture(root):
            fx = FixtureHostProbe(root)
            return fx, fx.meta.get("code_roots", [fx.home()])  # type: ignore[return-value]
        return LocalHostProbe(mode="folder"), [str(Path(root).resolve())]
    return LocalHostProbe(mode="host"), [os.getcwd()]
