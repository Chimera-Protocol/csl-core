"""
The Venom workspace: the folder where `cslcore setup` runs.

    <workspace>/policies/                active policies and mappings
    <workspace>/.csl/venom/state.json    setup progress, per-agent status, modes
    <workspace>/.csl/venom/inventory/    scan snapshots
    <workspace>/.csl/venom/reports/      reports
    <workspace>/.csl/venom/drafts/       policy drafts
    <workspace>/.csl/venom/exemptions.yaml
    <workspace>/.csl/venom/decisions/    decision logs

This module and probe.py are the only places Venom touches the filesystem. Every write
here is called from a path that the operator confirmed (or from --yes / non-destructive
bookkeeping such as saving a scan snapshot and report).
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import exemptions as ex
from .model import Exemption


class Workspace:
    def __init__(self, root: str | os.PathLike, plan_only: bool = False) -> None:
        self.root = Path(root).resolve()
        self.plan_only = plan_only
        self.planned: List[str] = []  # writes skipped because of --plan-only
        self._appendable: set = set()  # log files already checked to be inside the workspace
        self._state_file = str(self.root / ".csl" / "venom" / "state.json")

    # layout -----------------------------------------------------------------
    @property
    def policies(self) -> Path: return self.root / "policies"
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        if path.parent == self.policies and path.suffix in (".csl", ".py"):
            self.bump()
        return path

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

    # state --------------------------------------------------------------------
    def load_state(self) -> Dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def mtime(path) -> Optional[float]:
        try:
            return Path(path).stat().st_mtime_ns / 1e9
        except OSError:
            return None

    def state_mtime(self) -> Optional[float]:
        try:
            return os.stat(self._state_file).st_mtime_ns / 1e9
        except OSError:
            return None

    def save_state(self, state: Dict[str, Any]) -> None:
        self.write_text(self.state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")

    def bump(self) -> None:
        """Signal running guards that a policy, mapping or binding changed (one stat per call to notice)."""
        if self.plan_only:
            return
        state = self.load_state()
        state["generation"] = int(state.get("generation", 0)) + 1
        self.save_state(state)

    def update_state(self, **changes: Any) -> Dict[str, Any]:
        state = self.load_state()
        state.update(changes)
        self.save_state(state)
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
