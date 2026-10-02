"""
L5 run history: how often each agent ran, from real evidence only.

No evidence means count None ("n/a"), never zero. Every RunStats names its source.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ..model import RunStats


def parse_window(spec: str) -> int:
    m = re.fullmatch(r"(\d+)\s*([dhw]?)", spec.strip())
    if not m:
        raise ValueError(f"invalid --since value '{spec}' (examples: 7d, 30d, 2w)")
    n, unit = int(m.group(1)), m.group(2) or "d"
    return max(1, {"d": n, "w": n * 7, "h": max(1, n // 24)}[unit])


def day_keys(now: datetime, days: int) -> List[str]:
    return [(now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d") for i in range(days)]


def stats_from_days(counts: Dict[str, int], now: datetime, days: int, source: str,
                    last_run: Optional[str] = None) -> RunStats:
    keys = day_keys(now, days)
    per_day = [counts.get(k, 0) for k in keys]
    if last_run is None:
        present = [k for k in keys if counts.get(k)]
        last_run = present[-1] if present else None
    return RunStats(count=sum(per_day), window=f"{days}d", last_run=last_run, per_day=per_day, source=source)


def journal_starts(probe, unit: str, now: datetime, days: int) -> Optional[RunStats]:
    r = probe.run(["journalctl", "-u", unit, "--since", f"-{days}d", "-o", "json", "--no-pager"], timeout=15)
    if r is None or r.returncode != 0:
        return None
    counts: Dict[str, int] = {}
    last = None
    for line in r.stdout.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        msg = str(rec.get("MESSAGE", ""))
        if not msg.startswith("Started "):
            continue
        ts = rec.get("__REALTIME_TIMESTAMP")
        if ts is None:
            continue
        dt = datetime.fromtimestamp(int(ts) / 1_000_000, timezone.utc)
        counts[dt.strftime("%Y-%m-%d")] = counts.get(dt.strftime("%Y-%m-%d"), 0) + 1
        last = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return stats_from_days(counts, now, days, f"journalctl -u {unit}", last)


_SYSLOG_CRON = re.compile(r"^(\w{3}\s+\d+\s+[\d:]+)\s.*CRON\[\d+\]:\s+\(([^)]+)\)\s+CMD\s+\((.*)\)\s*$")
_ISO_CRON = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+\S*)\s.*CRON\[\d+\]:\s+\(([^)]+)\)\s+CMD\s+\((.*)\)\s*$")


def cron_log_runs(probe, command_hint: str, now: datetime, days: int) -> Optional[RunStats]:
    for path in ("/var/log/cron", "/var/log/syslog", "/var/log/cron.log"):
        if not probe.exists(path):
            continue
        text = probe.read_text(path, limit=20_000_000)
        if text is None:
            continue
        counts: Dict[str, int] = {}
        for line in text.splitlines():
            m = _ISO_CRON.match(line) or _SYSLOG_CRON.match(line)
            if not m or command_hint not in m.group(3):
                continue
            raw = m.group(1)
            try:
                dt = datetime.fromisoformat(raw) if "T" in raw else datetime.strptime(f"{now.year} {raw}", "%Y %b %d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if (now - dt).days < days:
                counts[dt.strftime("%Y-%m-%d")] = counts.get(dt.strftime("%Y-%m-%d"), 0) + 1
        return stats_from_days(counts, now, days, path)
    return None


def decision_log_runs(probe, path: str, now: datetime, days: int) -> Optional[RunStats]:
    if not probe.exists(path):
        return None
    text = probe.read_text(path, limit=50_000_000)
    if text is None:
        return None
    counts: Dict[str, int] = {}
    for line in text.splitlines():
        try:
            ts = json.loads(line).get("ts")
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError, AttributeError):
            continue
        if (now - dt).days < days:
            counts[dt.strftime("%Y-%m-%d")] = counts.get(dt.strftime("%Y-%m-%d"), 0) + 1
    return stats_from_days(counts, now, days, "csl decision log")
