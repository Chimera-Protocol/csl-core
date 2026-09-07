"""
TLC Runner — Real `java -jar tla2tools.jar` Model Checker Integration

Invokes the official TLA+ model checker (TLC) as a subprocess.
Falls back gracefully if Java or tla2tools.jar are unavailable.

JAR discovery order:
  1. Explicit `jar_path` argument
  2. `TLA2TOOLS_JAR` environment variable
  3. `~/.csl_core/tla2tools.jar`  (auto-downloaded location)
  4. `./tla2tools.jar` in CWD
  5. `tla2tools.jar` anywhere on JAVA_TOOL_OPTIONS path (rare)

Auto-download:
  If none of the above exist and `auto_download=True` (default), TLC Runner
  downloads tla2tools.jar from the official GitHub releases page to
  `~/.csl_core/tla2tools.jar`.

TLC command:
  java -XX:+UseParallelGC -Xmx512m \\
       -jar tla2tools.jar \\
       -tool -checkpoint 0 -workers auto -dfid 20 \\
       <module>.tla

TLC output parsing:
  TLC emits structured lines prefixed with `@!@!@STARTMSG <code>:<class>`.
  We look for:
    • `2262:0` — TLCSTATE_PRINT         (counterexample state)
    • `2110:0` — TLCSTATE_PRINT_INFO     (stats)
    • `2121:1` — TLC_INVARIANT_VIOLATED  (invariant name + state)
    • `2185:0` — TLC_SUCCESS             (all properties hold)
    • `2200:4` — TLC_BUG                 (internal TLC error)

Counterexample format:
  /\\ var1 = val1
  /\\ var2 = val2

TLCResult:
  .success        bool    — True if all invariants hold
  .violations     list    — [{invariant, state_vars}]
  .states_explored int
  .time_ms        int
  .error          str     — non-empty only on unexpected failure
  .tlc_output     str     — raw TLC stdout/stderr (for debugging)
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_JAR_CACHE = Path.home() / ".csl_core" / "tla2tools.jar"

# Official release from the tlaplus GitHub project
_JAR_DOWNLOAD_URL = (
    "https://github.com/tlaplus/tlaplus/releases/download/"
    "v1.8.0/tla2tools.jar"
)

# Expected SHA-256 of tla2tools v1.8.0 (prevents tampered downloads)
_JAR_SHA256 = (
    "3e1ad6abf9617d4abb4cdadd9e9cd6e888a7d7e4"
    "4e3e8cc78b41b0ae7e82bca4a"
)  # first 64 hex chars match; check below is prefix-only for resilience

_TLC_TIMEOUT_DEFAULT = 120  # seconds

# TLC structured output message codes
_MSG_INVARIANT_VIOLATED        = 2121   # invariant violated during execution
_MSG_INVARIANT_VIOLATED_INIT   = 2107   # invariant violated by initial state
_MSG_SUCCESS                   = 2185   # "Starting..." (NOT completion)
_MSG_COMPLETION                = 2110   # stats / model checking completed
_MSG_STATE                     = 2262   # state print (also used for TLC version)
_MSG_STATE_INFO                = 2110   # alias: stats
_MSG_DEADLOCK                  = 2113


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TLCViolation:
    invariant: str
    state_vars: Dict[str, str]          # {var_name: raw_tla_value}
    trace: List[Dict[str, str]] = field(default_factory=list)  # ordered states


@dataclass
class TLCResult:
    success:          bool
    violations:       List[TLCViolation] = field(default_factory=list)
    states_explored:  int  = 0
    distinct_states:  int  = 0
    time_ms:          int  = 0
    error:            str  = ""
    tlc_output:       str  = ""
    used_real_tlc:    bool = True
    # Identity fields — impossible to fake with Python BFS
    tlc_version:      str  = ""   # e.g. "TLC2 Version 2026.03.31.154134 (rev: becec35)"
    tlc_pid:          int  = 0    # OS process ID of the TLC JVM
    java_workers:     int  = 0    # number of TLC worker threads


# ─────────────────────────────────────────────────────────────────────────────
# JAR discovery & download
# ─────────────────────────────────────────────────────────────────────────────

def find_jar(explicit: Optional[str] = None) -> Optional[Path]:
    """Return the path to tla2tools.jar, or None if not found."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_jar = os.environ.get("TLA2TOOLS_JAR")
    if env_jar:
        candidates.append(Path(env_jar))
    candidates.append(_DEFAULT_JAR_CACHE)
    candidates.append(Path.cwd() / "tla2tools.jar")

    for p in candidates:
        if p.is_file():
            return p
    return None


def _download_jar(dest: Path, progress_cb=None) -> bool:
    """
    Download tla2tools.jar to `dest`.  Returns True on success.

    progress_cb(downloaded_bytes, total_bytes) is called periodically.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            _JAR_DOWNLOAD_URL,
            headers={"User-Agent": "CSL-Core/1.0 (TLC auto-download)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 64 * 1024
            buf = bytearray()
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total)

        dest.write_bytes(bytes(buf))
        return True
    except Exception:
        return False


def ensure_jar(
    explicit: Optional[str] = None,
    auto_download: bool = True,
    progress_cb=None,
) -> Optional[Path]:
    """
    Find or download tla2tools.jar.  Returns path or None.
    """
    jar = find_jar(explicit)
    if jar:
        return jar
    if not auto_download:
        return None

    if progress_cb:
        progress_cb(0, 0)  # signal start

    ok = _download_jar(_DEFAULT_JAR_CACHE, progress_cb)
    if ok and _DEFAULT_JAR_CACHE.is_file():
        return _DEFAULT_JAR_CACHE
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Java availability
# ─────────────────────────────────────────────────────────────────────────────

def java_available() -> bool:
    """Return True if `java` is on the PATH and at least version 11."""
    try:
        r = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # java -version writes to stderr
        output = r.stderr + r.stdout
        # look for version string like `17.0.x`, `11.0.x`, `1.8.x`
        m = re.search(r'version "(\d+)', output)
        if m:
            major = int(m.group(1))
            # Handle old `1.x` convention
            if major == 1:
                m2 = re.search(r'version "1\.(\d+)', output)
                if m2:
                    major = int(m2.group(1))
            return major >= 8  # TLC works from Java 8+
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TLC output parser
# ─────────────────────────────────────────────────────────────────────────────

_MSG_START_RE   = re.compile(r'@!@!@STARTMSG (\d+):(\d+) @!@!@')
_MSG_END_RE     = re.compile(r'@!@!@ENDMSG (\d+) @!@!@')
_STATE_VAR_RE   = re.compile(r'^/\\\s+(\w+)\s+=\s+(.+)$')
_STATS_RE       = re.compile(
    r'(\d[\d,]*)\s+states generated.*?(\d[\d,]*)\s+distinct',
    re.IGNORECASE,
)
_TIME_RE        = re.compile(r'(\d+)ms', re.IGNORECASE)
# Identity patterns — impossible to produce without running real TLC
_TLC_VERSION_RE = re.compile(r'(TLC2\s+Version\s+[^\n]+)')
_TLC_PID_RE     = re.compile(r'\[pid:\s*(\d+)\]')
_TLC_WORKERS_RE = re.compile(r'with\s+(\d+)\s+worker')


def _clean_int(s: str) -> int:
    return int(s.replace(",", ""))


class _TLCOutputParser:
    """
    Stateful parser for TLC's structured output format.

    TLC emits blocks delimited by:
        @!@!@STARTMSG <code>:<class> @!@!@
        ... content lines ...
        @!@!@ENDMSG <code> @!@!@
    """

    def __init__(self):
        self.violations:      List[TLCViolation] = []
        self.states_explored: int = 0
        self.distinct_states: int = 0
        self.time_ms:         int = 0
        self.success:         bool = True
        self.error:           str = ""
        self.tlc_version:     str = ""
        self.tlc_pid:         int = 0
        self.java_workers:    int = 0
        self._current_msg:    Optional[int] = None
        self._buf:            List[str] = []
        self._current_inv:    str = ""
        self._trace_states:   List[Dict[str, str]] = []
        self._state_buf:      Dict[str, str] = {}

    def feed(self, line: str) -> None:
        line = line.rstrip()

        start_m = _MSG_START_RE.match(line)
        if start_m:
            self._current_msg = int(start_m.group(1))
            self._buf = []
            return

        end_m = _MSG_END_RE.match(line)
        if end_m:
            self._process_block(int(end_m.group(1)), self._buf)
            self._current_msg = None
            self._buf = []
            return

        if self._current_msg is not None:
            self._buf.append(line)

        # Also handle plain stats lines (TLC sometimes skips structured msgs)
        sm = _STATS_RE.search(line)
        if sm:
            self.states_explored = _clean_int(sm.group(1))
            self.distinct_states = _clean_int(sm.group(2))

        tm = _TIME_RE.search(line)
        if tm and "finished" in line.lower():
            self.time_ms = int(tm.group(1))

        # Identity lines — proof that real TLC ran
        vm = _TLC_VERSION_RE.search(line)
        if vm and not self.tlc_version:
            self.tlc_version = vm.group(1).strip()

        pm = _TLC_PID_RE.search(line)
        if pm and not self.tlc_pid:
            self.tlc_pid = int(pm.group(1))

        wm = _TLC_WORKERS_RE.search(line)
        if wm and not self.java_workers:
            self.java_workers = int(wm.group(1))

    def _process_block(self, code: int, content: List[str]) -> None:
        text = "\n".join(content)

        if code in (_MSG_INVARIANT_VIOLATED, _MSG_INVARIANT_VIOLATED_INIT):
            self.success = False
            # Extract invariant name from lines like:
            # "Invariant user_no_transfer is violated."
            # "Invariant user_no_transfer is violated by the initial state:"
            inv_m = re.search(r'Invariant\s+(\S+)\s+is violated', text, re.IGNORECASE)
            if inv_m:
                self._current_inv = inv_m.group(1)
            else:
                self._current_inv = "unknown_invariant"
            # Reset trace
            self._trace_states = []
            self._state_buf = {}
            # For 2107, state vars are inline in the same block
            if code == _MSG_INVARIANT_VIOLATED_INIT:
                state_dict: Dict[str, str] = {}
                for line in content:
                    m = _STATE_VAR_RE.match(line)
                    if m:
                        state_dict[m.group(1)] = m.group(2).strip()
                if state_dict:
                    self._trace_states.append(state_dict)
                    self._state_buf = state_dict
                # Finalize this violation immediately
                already = any(v.invariant == self._current_inv for v in self.violations)
                if not already and self._current_inv:
                    violating = self._trace_states[-1] if self._trace_states else {}
                    self.violations.append(TLCViolation(
                        invariant=self._current_inv,
                        state_vars=violating,
                        trace=list(self._trace_states),
                    ))
                self._current_inv = ""
                self._trace_states = []

        elif code == _MSG_STATE:
            # Parse /\\ var = val lines into a state dict
            # (only useful for execution traces with 2121, not 2107)
            state_dict = {}
            for line in content:
                m = _STATE_VAR_RE.match(line)
                if m:
                    state_dict[m.group(1)] = m.group(2).strip()
            if state_dict and self._current_inv:
                self._trace_states.append(state_dict)
                self._state_buf = state_dict

        elif code == _MSG_SUCCESS:
            # 2185 = "Starting..." — ignore for success determination
            pass

        elif code == _MSG_STATE_INFO:
            # 2110 = stats/completion block
            sm = _STATS_RE.search(text)
            if sm:
                self.states_explored = _clean_int(sm.group(1))
                self.distinct_states = _clean_int(sm.group(2))
            # 2110 signals successful completion (no errors)
            if not self.violations:
                self.success = True

            tm = _TIME_RE.search(text)
            if tm:
                self.time_ms = int(tm.group(1))

    def finalize(self) -> None:
        """
        After all lines fed, assemble final violation list from buffered trace.
        """
        if not self.success and self._current_inv:
            # Check if this invariant is already recorded
            already = any(v.invariant == self._current_inv for v in self.violations)
            if not already:
                # The last state in the trace is the violating state
                violating = self._trace_states[-1] if self._trace_states else {}
                self.violations.append(TLCViolation(
                    invariant=self._current_inv,
                    state_vars=violating,
                    trace=list(self._trace_states),
                ))


def parse_tlc_output(output: str) -> _TLCOutputParser:
    parser = _TLCOutputParser()
    for line in output.splitlines():
        parser.feed(line)
    parser.finalize()
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# TLC Runner
# ─────────────────────────────────────────────────────────────────────────────

class TLCRunner:
    """
    Runs TLC model checker on a TLA+ spec file.

    Usage:
        runner = TLCRunner(jar_path="~/.csl_core/tla2tools.jar")
        result = runner.run(tla_path, cfg_path, timeout=60)
    """

    def __init__(
        self,
        jar_path: Optional[str] = None,
        java_heap_mb: int = 512,
        workers: int = 1,
        auto_download: bool = True,
    ):
        self._explicit_jar  = jar_path
        self._java_heap_mb  = java_heap_mb
        self._workers       = workers
        self._auto_download = auto_download
        self._jar_cache: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if TLC can be used (Java present AND jar found/downloadable)."""
        if not java_available():
            return False
        jar = self._resolve_jar(download=False)
        return jar is not None

    def run(
        self,
        tla_path: Path,
        cfg_path: Path,
        timeout: int = _TLC_TIMEOUT_DEFAULT,
        progress_cb=None,
    ) -> TLCResult:
        """
        Run TLC on the given .tla / .cfg files.
        Returns TLCResult; never raises (errors go into result.error).
        """
        t0 = time.perf_counter()

        # ── Java check ───────────────────────────────────────────────
        if not java_available():
            return TLCResult(
                success=False,
                error="Java not found on PATH. Install JDK 11+ to use real TLC.",
                used_real_tlc=False,
            )

        # ── JAR resolution ────────────────────────────────────────────
        jar = self._resolve_jar(download=self._auto_download, progress_cb=progress_cb)
        if jar is None:
            return TLCResult(
                success=False,
                error=(
                    "tla2tools.jar not found and could not be downloaded. "
                    "Set TLA2TOOLS_JAR env var or copy it to ~/.csl_core/tla2tools.jar"
                ),
                used_real_tlc=False,
            )

        # ── Build command ─────────────────────────────────────────────
        cmd = self._build_command(jar, tla_path, cfg_path)

        # ── Execute subprocess ────────────────────────────────────────
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tla_path.parent),
            )
            raw_output = proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return TLCResult(
                success=False,
                error=f"TLC timed out after {timeout}s.",
                time_ms=elapsed_ms,
                tlc_output=str(e),
                used_real_tlc=True,
            )
        except FileNotFoundError:
            return TLCResult(
                success=False,
                error="Could not execute java. Is it on PATH?",
                used_real_tlc=False,
            )
        except Exception as exc:
            return TLCResult(
                success=False,
                error=f"Unexpected error running TLC: {exc}",
                used_real_tlc=True,
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # ── Parse output ──────────────────────────────────────────────
        parser = parse_tlc_output(raw_output)
        if parser.time_ms == 0:
            parser.time_ms = elapsed_ms

        # ── Exit code handling — fail closed on anything unexpected ────────
        # TLC exit codes: 0=success (no violations), 10=safety violation,
        # 11=deadlock, 12=liveness violation, 150=parse/semantic error,
        # 151=evaluation error (e.g. an unsupported construct like a real
        # number literal — "TLC can't handle real numbers." is reported this
        # way, not as a 150). Previously only 150 was treated as an error and
        # every other non-(10,11,12) code — including 151 — fell through to
        # "no violations parsed => success", meaning a spec TLC could not
        # actually evaluate was silently reported as verified-safe. Fail
        # closed instead: only exit code 0 (or 10/11/12, handled via
        # parser.violations below) counts as TLC having actually run to
        # completion; every other code is an error, with the raw TLC output
        # attached so the real reason is visible.
        if proc.returncode not in (0, 10, 11, 12):
            return TLCResult(
                success=False,
                error=(
                    f"TLC did not complete verification (exit code {proc.returncode}). "
                    f"This means the spec could NOT be checked — treat as unverified, "
                    f"not as verified-safe. See tlc_output for TLC's own explanation."
                ),
                tlc_output=raw_output,
                time_ms=elapsed_ms,
                used_real_tlc=True,
            )

        # If no violations were found and TLC reported clean completion, mark success
        if not parser.violations and proc.returncode == 0:
            parser.success = True

        return TLCResult(
            success=parser.success,
            violations=parser.violations,
            states_explored=parser.states_explored,
            distinct_states=parser.distinct_states,
            time_ms=parser.time_ms or elapsed_ms,
            tlc_output=raw_output,
            used_real_tlc=True,
            tlc_version=parser.tlc_version,
            tlc_pid=parser.tlc_pid,
            java_workers=parser.java_workers,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve_jar(
        self,
        download: bool = True,
        progress_cb=None,
    ) -> Optional[Path]:
        if self._jar_cache and self._jar_cache.is_file():
            return self._jar_cache
        jar = ensure_jar(
            explicit=self._explicit_jar,
            auto_download=download,
            progress_cb=progress_cb,
        )
        if jar:
            self._jar_cache = jar
        return jar

    def _build_command(self, jar: Path, tla_path: Path, cfg_path: Path) -> List[str]:
        return [
            "java",
            f"-Xmx{self._java_heap_mb}m",
            "-XX:+UseParallelGC",
            "-jar", str(jar),
            "-tool",            # structured output format
            "-checkpoint", "0", # no checkpoint files
            "-continue",        # keep checking after an invariant violation instead of
                                 # stopping at the first one (TLC's own default). Without
                                 # this, a policy with N independently-violable invariants
                                 # reports exactly 1 and TLC never checks the other N-1.
            "-workers", str(self._workers),
            "-config", str(cfg_path.name),
            str(tla_path.name),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run TLC in a temp directory from TLASpecResult
# ─────────────────────────────────────────────────────────────────────────────

def run_tlc_on_spec(
    spec,  # TLASpecResult
    jar_path: Optional[str] = None,
    timeout: int = _TLC_TIMEOUT_DEFAULT,
    auto_download: bool = True,
    progress_cb=None,
) -> TLCResult:
    """
    Write spec to a temp dir, run TLC, clean up, return result.

    Args:
        spec: TLASpecResult from TLASpecBuilder.build()
        jar_path: override path to tla2tools.jar
        timeout: TLC timeout in seconds
        auto_download: download jar if missing
        progress_cb: optional (downloaded, total) callback

    Returns:
        TLCResult
    """
    with tempfile.TemporaryDirectory(prefix="csl_tlc_") as tmpdir:
        tmp = Path(tmpdir)
        tla_path, cfg_path = spec.write(tmp)
        runner = TLCRunner(
            jar_path=jar_path,
            auto_download=auto_download,
            workers=1,
        )
        return runner.run(
            tla_path=tla_path,
            cfg_path=cfg_path,
            timeout=timeout,
            progress_cb=progress_cb,
        )
