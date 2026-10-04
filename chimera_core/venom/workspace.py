"""
The Venom workspace: the folder where `cslcore setup` runs.

    <workspace>/policies/                active policies and mappings
    <workspace>/.csl/venom/state.json    setup progress, per-agent status, modes
    <workspace>/.csl/venom/inventory/    scan snapshots
    <workspace>/.csl/venom/reports/      reports
    <workspace>/.csl/venom/drafts/       policy drafts
    <workspace>/.csl/venom/exemptions.yaml
    <workspace>/.csl/venom/decisions/    decision logs
    <workspace>/.csl/venom/wire/         copies of files `cslcore wire` changed, to undo it

This module and probe.py are the only places Venom touches the filesystem. Every write
here is called from a path that the operator confirmed (or from --yes / non-destructive
bookkeeping such as saving a scan snapshot and report). Writes stay inside the workspace,
with one exception: `change_file`, which applies a change the operator saw as a diff and
confirmed (wiring a guard into an agent), keeps a copy of the file first and refuses when the
file changed after the diff was shown.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import exemptions as ex
from .model import Exemption


def has_workspace(folder: Path) -> bool:
    return (folder / ".csl" / "venom").is_dir()


def locate(given: Optional[str] = None, near: Optional[str] = None) -> Path:
    """The workspace a guard uses, wherever the agent runs (another machine, CI, a container):

        1. CSL_WORKSPACE, when it is set (the operator's override for a deployment)
        2. `given`: absolute, or relative to the file the guard is created in (`near`)
        3. the nearest folder above that file (or the current folder) that has a .csl workspace
        4. `given` as it is, or the current folder (the guard then refuses calls: fail closed)

    Lines written before 0.6.9 pass an absolute path: it is used when it holds a workspace."""
    env = os.environ.get("CSL_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    base = Path(near).resolve().parent if near else Path.cwd()
    candidate = None
    if given:
        g = Path(given).expanduser()
        candidate = (g if g.is_absolute() else base / g).resolve()
        if has_workspace(candidate):
            return candidate
    for folder in [base, *base.parents]:
        if has_workspace(folder):
            return folder
    return candidate or Path.cwd().resolve()


def scope_root_paths(roots: List[str], near: str) -> List[str]:
    """Scope roots written relative to a generated mapping file, made absolute where it runs."""
    here = Path(near).resolve().parent
    return [r if os.path.isabs(r) else str((here / r).resolve()) for r in roots]


class StateUnreadable(Exception):
    """state.json exists but cannot be read or parsed (a guard then keeps what it knew)."""


class LockTimeout(Exception):
    """The state lock was not free within the time asked for."""


def _lock_file(fh, path: Path, timeout: Optional[float]) -> object:
    """Take the lock between writers, on any OS; returns what _unlock_file needs.

    POSIX: flock. Windows: msvcrt.locking on the first byte. Neither: a lock file made with
    O_CREAT | O_EXCL (a stale one, older than a minute, is taken over). The state needs a local
    disk: flock is not reliable on NFS and some container volumes."""
    import time as _time

    deadline = None if timeout is None else _time.monotonic() + timeout
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | (fcntl.LOCK_NB if deadline is not None else 0))
                return ("flock", fh)
            except BlockingIOError:
                if _time.monotonic() >= deadline:
                    raise LockTimeout(str(path))
                _time.sleep(0.01)
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    if msvcrt is not None:
        while True:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return ("msvcrt", fh)
            except OSError:
                if deadline is not None and _time.monotonic() >= deadline:
                    raise LockTimeout(str(path))
                _time.sleep(0.01)
    marker = path.with_suffix(".held")
    while True:
        try:
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return ("file", marker)
        except FileExistsError:
            try:
                if _time.time() - marker.stat().st_mtime > 60:
                    marker.unlink()  # its writer is gone
                    continue
            except OSError:
                continue
            if deadline is not None and _time.monotonic() >= deadline:
                raise LockTimeout(str(path))
            _time.sleep(0.01)


def _unlock_file(held) -> None:
    kind, obj = held
    if kind == "flock":
        import fcntl
        fcntl.flock(obj.fileno(), fcntl.LOCK_UN)
    elif kind == "msvcrt":
        import msvcrt
        obj.seek(0)
        msvcrt.locking(obj.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        try:
            obj.unlink()
        except OSError:
            pass


class Workspace:
    def __init__(self, root: str | os.PathLike, plan_only: bool = False) -> None:
        self.root = Path(root).resolve()
        self._loaded: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self.plan_only = plan_only
        self.planned: List[str] = []  # writes skipped because of --plan-only
        self._appendable: set = set()  # log files already checked to be inside the workspace
        self._state_file = str(self.root / ".csl" / "venom" / "state.json")

    # layout -----------------------------------------------------------------
    @property
    def policies(self) -> Path:
        """Active policies and their mappings: .csl/policies (0.6.9). A workspace made before keeps
        policies/ in its root, recognised by a generated mapping in it, and goes on using it."""
        legacy = self.root / "policies"
        if not (self.root / ".csl" / "policies").exists() and legacy.is_dir() and any(legacy.glob("*_mapping.py")):
            return legacy
        return self.root / ".csl" / "policies"
    @property
    def venom(self) -> Path: return self.root / ".csl" / "venom"
    @property
    def drafts(self) -> Path: return self.venom / "drafts"
    @property
    def reports(self) -> Path: return self.venom / "reports"
    @property
    def inventory_dir(self) -> Path: return self.venom / "inventory"
    @property
    def decisions(self) -> Path: return self.venom / "decisions"
    @property
    def state_path(self) -> Path: return self.venom / "state.json"
    @property
    def exemptions_path(self) -> Path: return self.venom / "exemptions.yaml"

    def exists(self) -> bool:
        return self.venom.is_dir()

    def rel(self, p: Path | str) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    # writing ----------------------------------------------------------------
    def write_text(self, path: Path, text: str) -> Optional[Path]:
        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise PermissionError(f"refusing to write outside the workspace: {path}")
        if self.plan_only:
            self.planned.append(self.rel(path))
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ignore_csl(path)
        # a temp file of its own (process, thread, random): two writers never share one, and the
        # rename makes the new content appear whole or not at all
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        if path.parent == self.policies and path.suffix in (".csl", ".py"):
            self.bump()
        return path

    def _ignore_csl(self, path: Path) -> None:
        """.csl/ is this machine's workspace (state, policies, logs): kept out of the repository by its
        own .gitignore. What a team commits is csl-limits.ini (cslcore apply makes .csl/ from it)."""
        csl = self.root / ".csl"
        ignore = csl / ".gitignore"
        if not ignore.exists() and Path(path).resolve().is_relative_to(csl.resolve() if csl.exists() else csl):
            try:
                ignore.write_text("# the workspace of this machine; commit csl-limits.ini instead\n*\n", encoding="utf-8")
            except OSError:
                pass

    def append_line(self, path: Path, line: str) -> None:
        """Append one line (decision logs). Always inside the workspace (checked once per file)."""
        key = str(path)
        if key not in self._appendable:
            p = Path(path)
            if not p.resolve().is_relative_to(self.root):
                raise PermissionError(f"refusing to write outside the workspace: {path}")
            p.parent.mkdir(parents=True, exist_ok=True)
            self._appendable.add(key)
        with open(key, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    def remove(self, path: Path) -> None:
        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise PermissionError(f"refusing to remove outside the workspace: {path}")
        if self.plan_only:
            self.planned.append(f"remove {self.rel(path)}")
            return
        if path.exists():
            path.unlink()

    # the one write outside the workspace ------------------------------------------
    @staticmethod
    def sha(text: Optional[str]) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def change_file(self, path: str, before_sha: str, after: str, backup_dir: Path) -> Optional[Dict[str, Any]]:
        """Apply a confirmed change to a file anywhere (an agent's code, its assistant settings).
        The current file must still be what the diff was made from (`before_sha`); a copy goes
        into `backup_dir` (inside the workspace) first. Returns what undo needs."""
        target = Path(path)
        current = self.read(target) if target.exists() else None
        if self.sha(current) != before_sha:
            raise RuntimeError(f"{path} changed after the diff was shown; nothing was written")
        if self.plan_only:
            self.planned.append(f"change {path}")
            return None
        copy = None
        if current is not None:
            copy = backup_dir / (str(target.resolve()).lstrip("/").replace("/", "__"))
            self.write_text(copy, current)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".csl-tmp")
        tmp.write_text(after, encoding="utf-8")
        tmp.replace(target)
        return {"path": str(target), "backup": self.rel(copy) if copy else None, "after_sha": self.sha(after)}

    def undo_change(self, record: Dict[str, Any]) -> str:
        """Put a changed file back as it was: 'restored', 'removed' (it did not exist before), or
        'skipped' when the file changed again since (it is then left alone)."""
        target = Path(record["path"])
        current = self.read(target) if target.exists() else None
        if self.sha(current) != record.get("after_sha"):
            return "skipped"
        if record.get("backup"):
            original = self.read(self.root / record["backup"])
            if original is None:
                return "skipped"
            tmp = target.with_name(target.name + ".csl-tmp")
            tmp.write_text(original, encoding="utf-8")
            tmp.replace(target)
            return "restored"
        target.unlink()
        return "removed"

    # state --------------------------------------------------------------------
    def load_state(self) -> Dict[str, Any]:
        try:
            state = self.load_state_strict()
        except StateUnreadable:
            return {}
        self._loaded[id(state)] = (state, copy.deepcopy(state))  # what this caller started from (save_state merges)
        if len(self._loaded) > 256:
            self._loaded.pop(next(iter(self._loaded)))
        return state

    def load_state_strict(self) -> Dict[str, Any]:
        """The state as it is on disk; {} when there is none yet; StateUnreadable when it cannot be
        read or parsed. Guards use this: an unreadable state never reads as "nothing set"."""
        try:
            text = self.state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as e:
            raise StateUnreadable(str(e)) from e
        try:
            data = json.loads(text)
        except ValueError as e:
            raise StateUnreadable(str(e)) from e
        if not isinstance(data, dict):
            raise StateUnreadable("not an object")
        return data

    @contextmanager
    def state_lock(self, timeout: Optional[float] = None):
        """One writer at a time for the workspace's state files, across threads and processes.
        Only writers take it: a guard reads without it (writes are atomic renames). `timeout`:
        give up with LockTimeout instead of waiting (a tool call never waits on the panel)."""
        if not self._thread_lock.acquire(timeout=-1 if timeout is None else timeout):
            raise LockTimeout(str(self.venom / "state.lock"))
        try:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            fh = held = None
            if not self.plan_only:
                lock = self.venom / "state.lock"
                lock.parent.mkdir(parents=True, exist_ok=True)
                self._ignore_csl(lock)
                fh = open(lock, "a+")
                try:
                    held = _lock_file(fh, lock, timeout)
                except BaseException:
                    fh.close()
                    raise
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                if held is not None:
                    _unlock_file(held)
                if fh is not None:
                    fh.close()
        finally:
            self._thread_lock.release()

    @contextmanager
    def transaction(self):
        """Read, change and write state.json as one step: the controls (modes, freezes, tool
        switches, exemptions) are written this way, so no concurrent write can undo them."""
        with self.state_lock():
            try:
                state = self.load_state_strict()
            except StateUnreadable:
                state = {}
            yield state
            self._write_state(state)

    @staticmethod
    def mtime(path) -> Optional[float]:
        try:
            return Path(path).stat().st_mtime_ns / 1e9
        except OSError:
            return None

    def state_mtime(self):
        """A version of state.json: changes with every write (time, inode, size), so a guard never
        misses one, also where file times are coarse (an atomic rename always makes a new inode)."""
        try:
            st = os.stat(self._state_file)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_ino, st.st_size)

    def save_state(self, state: Dict[str, Any]) -> None:
        """Write the state; a key this caller did not change keeps what is on disk now, so a write
        that read the state earlier never undoes a concurrent one (a freeze, a mode, a limit)."""
        entry = self._loaded.pop(id(state), None)
        base = entry[1] if entry is not None and entry[0] is state else None
        with self.state_lock():
            if base is not None:
                try:
                    disk = self.load_state_strict()
                except StateUnreadable:
                    disk = None
                if disk is not None:
                    for k in set(disk) | set(state):
                        if state.get(k) == base.get(k) and disk.get(k) != base.get(k):
                            if k in disk:
                                state[k] = copy.deepcopy(disk[k])
                            else:
                                state.pop(k, None)
            self._write_state(state)

    def _write_state(self, state: Dict[str, Any]) -> None:
        self.write_text(self.state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")

    def bump(self) -> None:
        """Signal running guards that a policy, mapping or binding changed (one stat per call to notice)."""
        if self.plan_only:
            return
        state = self.load_state()
        state["generation"] = int(state.get("generation", 0)) + 1
        self.save_state(state)

    def update_state(self, **changes: Any) -> Dict[str, Any]:
        with self.transaction() as state:
            state.update(changes)
        return state

    # policies -------------------------------------------------------------------
    def policy_items(self) -> List[Tuple[str, str, str]]:
        out = []
        for folder, status in ((self.policies, "active"), (self.drafts, "draft")):
            if folder.is_dir():
                for p in sorted(folder.glob("*.csl")):
                    try:
                        out.append((str(p), p.read_text(encoding="utf-8"), status))
                    except OSError:
                        continue
        return out

    def read(self, path: Path | str) -> Optional[str]:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            return None

    # exemptions -------------------------------------------------------------------
    def load_exemptions(self) -> List[Exemption]:
        text = self.read(self.exemptions_path)
        return ex.parse(text) if text else []

    def save_exemptions(self, items: List[Exemption]) -> None:
        self.write_text(self.exemptions_path, ex.dump(items))

    # snapshots and reports ----------------------------------------------------------
    def latest_inventory(self) -> Optional[Dict[str, Any]]:
        p = self.inventory_dir / "latest.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_inventory(self, data: Dict[str, Any], stamp: str) -> None:
        text = json.dumps(data, indent=1, sort_keys=True) + "\n"
        self.write_text(self.inventory_dir / f"inventory-{stamp}.json", text)
        self.write_text(self.inventory_dir / "latest.json", text)

    def decision_logs(self) -> List[Path]:
        return sorted(self.decisions.glob("*.jsonl")) if self.decisions.is_dir() else []

    def read_new(self, path: Path, offset: int) -> Tuple[List[str], int]:
        """Complete lines appended to a log since `offset` (a partial last line is left for later)."""
        try:
            with open(path, "rb") as f:
                size = f.seek(0, 2)
                if size < offset:  # truncated or rotated
                    offset = 0
                f.seek(offset)
                data = f.read(8_000_000)
        except OSError:
            return [], offset
        end = data.rfind(b"\n")
        if end < 0:
            return [], offset
        chunk = data[: end + 1]
        return chunk.decode("utf-8", errors="replace").splitlines(), offset + len(chunk)

    def decision_log(self, agent_id: str) -> Path:
        safe = agent_id.replace("/", "_").replace(":", "_")
        return self.decisions / f"{safe}.jsonl"

    # mapping modules ------------------------------------------------------------------
    def load_module(self, path: Path):
        """Import an operator's mapping module (generated or hand-written) for the mapping test.
        Only files the operator points at are imported; discovered agent code never is."""
        import importlib.util

        path = Path(path).resolve()
        spec = importlib.util.spec_from_file_location(f"_venom_mapping_{abs(hash(str(path)))}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # editor -------------------------------------------------------------------------
    def editor(self) -> str:
        return os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"

    def svg_to_png(self, svg: Path, width: int = 2000) -> Optional[Path]:
        """Convert a workspace SVG to PNG with cairosvg or rsvg-convert when one is installed."""
        import shutil
        import subprocess

        svg = Path(svg)
        if not svg.resolve().is_relative_to(self.root):
            raise PermissionError(f"refusing to write outside the workspace: {svg}")
        png = svg.with_suffix(".png")
        try:
            import cairosvg  # type: ignore

            cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width)
            return png
        except Exception:
            pass
        tool = shutil.which("rsvg-convert")
        if tool:
            try:
                subprocess.run([tool, "-w", str(width), "-o", str(png), str(svg)], check=True, capture_output=True, timeout=60)
                return png
            except Exception:
                return None
        return None

    def open_in_editor(self, path: Path) -> int:
        """Open a workspace file in the operator's own editor (edit is always explicit)."""
        import shlex
        import subprocess

        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise PermissionError(f"refusing to edit outside the workspace: {path}")
        return subprocess.call(shlex.split(self.editor()) + [str(path)])

    # the operator's own programs (assistant CLI) ------------------------------------------
    @staticmethod
    def find_program(name: str) -> Optional[str]:
        import shutil
        return shutil.which(name)

    def run_program(self, argv: List[str], timeout: float = 300.0, env: Optional[Dict[str, str]] = None) -> int:
        """Run an operator-installed program (e.g. `claude -p`) in the workspace, after the operator agreed."""
        import subprocess

        try:
            r = subprocess.run(argv, cwd=str(self.root), timeout=timeout, env={**os.environ, **(env or {})},
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return r.returncode
        except (OSError, subprocess.TimeoutExpired):
            return -1

    def today(self) -> date:
        return date.today()
