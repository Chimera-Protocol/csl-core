"""
L3 trigger layer: what starts each agent.

Sources: user crontab, /etc/cron.d, systemd units and timers, launchd plists, and webhook
routes found by the code layer. A source that is not available is reported, never fatal.
"""

from __future__ import annotations

import plistlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, List, Optional

MESSAGING = re.compile(r"(sms|twilio|whatsapp|telegram|slack/events|slack|discord|messag|webhook/chat|teams)", re.I)
EMAIL = re.compile(r"(inbound-?email|email|mail|sendgrid/inbound|mailgun)", re.I)


def route_trigger_type(path: str, handler: str = "") -> str:
    text = f"{path} {handler}"
    if MESSAGING.search(text):
        return "messaging"
    if EMAIL.search(text):
        return "email"
    return "inbound_http"


@dataclass
class TriggerRecord:
    type: str  # time | inbound_http | messaging | email | manual | boot
    schedule: Optional[str]
    command: str
    source: str
    unit: Optional[str] = None
    run_as: Optional[str] = None


@dataclass
class TriggerScan:
    records: List[TriggerRecord] = field(default_factory=list)
    unavailable: Dict[str, str] = field(default_factory=dict)
    sources_ok: List[str] = field(default_factory=list)


_CRON_LINE = re.compile(r"^\s*((?:@\w+)|(?:\S+\s+\S+\s+\S+\s+\S+\s+\S+))\s+(.+)$")


def parse_crontab(text: str, source: str, system: bool = False) -> List[TriggerRecord]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or re.match(r"^[A-Z_]+=", s):
            continue
        m = _CRON_LINE.match(s)
        if not m:
            continue
        sched, rest = m.group(1), m.group(2)
        user = None
        if system:
            parts = rest.split(None, 1)
            if len(parts) == 2:
                user, rest = parts
        out.append(TriggerRecord("boot" if sched == "@reboot" else "time", sched, rest.strip(), source, run_as=user))
    return out


def _unit_exec(text: str) -> Optional[str]:
    m = re.search(r"(?m)^\s*ExecStart\s*=\s*[-@+!:]*(.+)$", text)
    return m.group(1).strip() if m else None


def _unit_user(text: str) -> Optional[str]:
    m = re.search(r"(?m)^\s*User\s*=\s*(\S+)", text)
    return m.group(1) if m else None


def scan_triggers(probe) -> TriggerScan:
    scan = TriggerScan()
    home = probe.home()

    # user crontab
    r = probe.run(["crontab", "-l"])
    if r is None:
        scan.unavailable["crontab"] = "crontab not available"
    else:
        scan.sources_ok.append("crontab")
        if r.returncode == 0:
            scan.records += parse_crontab(r.stdout, "crontab -l")
    for d in ("/etc/cron.d",):
        if probe.is_dir(d):
            for name in probe.list_dir(d):
                text = probe.read_text(f"{d}/{name}")
                if text:
                    scan.records += parse_crontab(text, f"{d}/{name}", system=True)
    if probe.exists("/etc/crontab"):
        text = probe.read_text("/etc/crontab")
        if text:
            scan.records += parse_crontab(text, "/etc/crontab", system=True)

    # systemd
    unit_dirs = ["/etc/systemd/system", f"{home}/.config/systemd/user", "/lib/systemd/system/../../../etc/systemd/system"]
    timers: Dict[str, str] = {}
    services: Dict[str, str] = {}
    found_systemd = False
    for d in unit_dirs[:2]:
        if not probe.is_dir(d):
            continue
        found_systemd = True
        for name in sorted(probe.list_dir(d)):
            p = f"{d}/{name}"
            if name.endswith(".service") or name.endswith(".timer"):
                text = probe.read_text(p)
                if text is None:
                    continue
                if name.endswith(".timer"):
                    m = re.search(r"(?m)^\s*(OnCalendar|OnUnitActiveSec|OnBootSec)\s*=\s*(.+)$", text)
                    unit = re.search(r"(?m)^\s*Unit\s*=\s*(\S+)", text)
                    target = unit.group(1) if unit else name[:-6] + ".service"
                    timers[target] = m.group(2).strip() if m else "timer"
                else:
                    services[name] = text
    for name, text in sorted(services.items()):
        cmd = _unit_exec(text)
        if not cmd:
            continue
        if name in timers:
            scan.records.append(TriggerRecord("time", timers[name], cmd, f"systemd:{name}", unit=name, run_as=_unit_user(text)))
        else:
            scan.records.append(TriggerRecord("boot", "service", cmd, f"systemd:{name}", unit=name, run_as=_unit_user(text)))
    if found_systemd:
        scan.sources_ok.append("systemd")
    elif probe.os_name() == "linux":
        scan.unavailable["systemd"] = "no unit folders readable"

    # launchd (macOS)
    if probe.os_name() == "macos":
        for d in (f"{home}/Library/LaunchAgents", "/Library/LaunchAgents", "/Library/LaunchDaemons"):
            if not probe.is_dir(d):
                continue
            for name in sorted(probe.list_dir(d)):
                if not name.endswith(".plist") or name.startswith("com.apple."):
                    continue
                p = f"{d}/{name}"
                text = probe.read_text(p)
                if text is None:
                    continue
                try:
                    data = plistlib.loads(text.encode("utf-8"))
                except Exception:
                    continue
                args = data.get("ProgramArguments") or ([data["Program"]] if data.get("Program") else [])
                if not args:
                    continue
                cmd = " ".join(str(a) for a in args)
                if data.get("StartInterval"):
                    kind, sched = "time", f"every {data['StartInterval']}s"
                elif data.get("StartCalendarInterval"):
                    kind, sched = "time", _calendar(data["StartCalendarInterval"])
                elif data.get("KeepAlive") or data.get("RunAtLoad"):
                    kind, sched = "boot", "at login" if "LaunchAgents" in d else "at boot"
                else:
                    kind, sched = "manual", None
                scan.records.append(TriggerRecord(kind, sched, cmd, f"launchd:{name}", unit=data.get("Label", name),
                                                  run_as=data.get("UserName")))
        scan.sources_ok.append("launchd")
    return scan


def _calendar(spec) -> str:
    specs = spec if isinstance(spec, list) else [spec]
    parts = []
    for s in specs[:3]:
        if isinstance(s, dict):
            h, m = s.get("Hour"), s.get("Minute", 0)
            wd = s.get("Weekday")
            t = f"{h:02d}:{m:02d}" if isinstance(h, int) else f"*:{m:02d}"
            parts.append(("weekly " if wd is not None else "daily ") + t)
    return ", ".join(parts) or "calendar"


def humanize_cron(expr: str) -> str:
    """Short label for common schedules: '0 6 * * *' -> 'daily 06:00'."""
    if expr.startswith("@"):
        return expr[1:]
    parts = expr.split()
    if len(parts) != 5:
        return expr
    mi, h, dom, mon, dow = parts
    if mi.isdigit() and h.isdigit() and dom == mon == dow == "*":
        return f"daily {int(h):02d}:{int(mi):02d}"
    if mi.isdigit() and h == "*" and dom == mon == dow == "*":
        return f"hourly :{int(mi):02d}"
    m = re.fullmatch(r"\*/(\d+)", mi)
    if m and h == dom == mon == dow == "*":
        return f"every {m.group(1)}m"
    if mi.isdigit() and h.isdigit() and dom == mon == "*" and dow.isdigit():
        return f"weekly {int(h):02d}:{int(mi):02d}"
    return expr


PY_PATH = re.compile(r"(/[^\s'\";|&]+\.py)\b")


def referenced_paths(command: str) -> List[str]:
    """Absolute .py paths and `cd <dir>` folders mentioned in a command line."""
    out = PY_PATH.findall(command)
    for m in re.finditer(r"\bcd\s+(/[^\s;&|]+)", command):
        out.append(m.group(1))
    for m in re.finditer(r"(?:--directory|--project|-C)\s+(/[^\s;&|]+)", command):
        out.append(m.group(1))
    return [str(PurePosixPath(p)) for p in out]
