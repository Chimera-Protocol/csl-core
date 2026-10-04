"""
Entity resolution: records from different layers that share an
entrypoint path, a project folder, a unit name or an assistant project merge into one
agent. Agent-like processes that match nothing become kind `unmanaged`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

from .analysis.risk import builtin_tools, classify, mcp_tools
from .layers.code import CodeFile, project_of
from .layers.config import AssistantRecord, ConfigScan, classify_credential
from .layers.history import cron_log_runs, journal_starts, stats_from_days
from .layers.processes import (
    AGENT_CLI, MCP_SERVER, PERMISSION_SKIP, RuntimeScan, assistant_product, is_loopback, process_cwd,
)
from .layers.triggers import TriggerScan, humanize_cron, referenced_paths, route_trigger_type
from .model import Agent, Credential, Evidence, PromptInfo, RunStats, Trigger


@dataclass
class Listener:
    pid: int
    args: str
    addresses: List[str]
    is_mcp: bool


@dataclass
class Resolution:
    agents: List[Agent] = field(default_factory=list)
    listeners: List[Listener] = field(default_factory=list)


def _short(path: str, home: str) -> str:
    return path.replace(home, "~", 1) if home and path.startswith(home) else path


def _unique_names(agents: List[Agent]) -> None:
    seen: Dict[str, int] = {}
    for a in agents:
        seen[a.display_name] = seen.get(a.display_name, 0) + 1
    for a in agents:
        if seen[a.display_name] > 1 and a.project:
            parent = PurePosixPath(a.project).parent.name
            a.display_name = f"{parent}/{a.display_name}"


def _within(path: str, folder: str) -> bool:
    folder = folder.rstrip("/")
    return path == folder or path.startswith(folder + "/")


HELPER_DIRS = {"tools", "tool", "utils", "util", "lib", "libs", "helpers", "helper", "common", "shared", "core",
               "functions", "skills", "actions", "integrations"}


def _rel_dir(project: str, path: str) -> str:
    rel = path[len(project.rstrip("/")) + 1:]
    return rel.rsplit("/", 1)[0] if "/" in rel else ""


def _home_dir(project: str, path: str) -> str:
    """The folder an agent file belongs to: its own, unless that is a helper folder (tools/, utils/, ...),
    whose tools belong to the folder above it."""
    d = _rel_dir(project, path)
    while d and d.rsplit("/", 1)[-1].lower() in HELPER_DIRS:
        d = d.rsplit("/", 1)[0] if "/" in d else ""
    return d


def _imports_folder(cf: CodeFile, folder: str) -> bool:
    """Whether a file imports a module from `folder` (by its dotted path, or relatively)."""
    dotted = folder.replace("/", ".")
    last = folder.rsplit("/", 1)[-1]
    from .layers.code import FRAMEWORK_IMPORTS

    frameworks = {prefix for prefix, _label in FRAMEWORK_IMPORTS}
    for m in cf.imports:
        if not m.startswith(".") and m.split(".", 1)[0] in frameworks:
            continue  # `from agents import Agent` is the OpenAI Agents SDK, not a folder named agents
        bare = m.lstrip(".")
        if bare == dotted or bare.startswith(dotted + ".") or (m.startswith(".") and (bare == last or bare.startswith(last + "."))):
            return True
    return False


def _definitions(cfs: List[CodeFile]) -> "OrderedDict[str, Tuple[List[str], CodeFile]]":
    """Explicit agent definitions in a project, each with the tools its list names (resolved across files)."""
    symbols: Dict[str, str] = {}
    lists: Dict[str, List[str]] = {}
    known = set()
    for f in cfs:
        symbols.update(f.symbols)
        lists.update(f.lists)
        known |= {t.name for t in f.tools}

    def expand(refs: List[str], depth: int = 0) -> List[str]:
        out: List[str] = []
        for r in refs:
            if r.startswith("*"):
                if depth < 3:
                    out += expand(lists.get(r[1:], []), depth + 1)
            else:
                t = symbols.get(r, r)
                if t in known and t not in out:
                    out.append(t)
        return out

    defs: "OrderedDict[str, Tuple[List[str], CodeFile]]" = OrderedDict()
    for f in cfs:
        for name, refs, _line in f.agent_defs:
            tools = expand(refs)
            if not tools:
                continue
            if name in defs:
                defs[name] = (defs[name][0] + [t for t in tools if t not in defs[name][0]], defs[name][1])
            else:
                defs[name] = (tools, f)
    return defs


def _units(project: str, cfs: List[CodeFile]) -> List[Tuple[str, str, List[CodeFile], Optional[set]]]:
    """The agents in one project: (id path, display name, files, the tools that are its own or None for all).

    Explicit definitions (Agent(name=..., tools=[...]) and the like) are agents of their own when there
    is more than one, or one beside tools it does not use. The rest is split by folder: tools in
    different folders are different agents, unless a folder is a helper (tools/, utils/, ...) or is
    imported from another agent's folder."""
    name = PurePosixPath(project).name or project
    defs = _definitions(cfs)
    claimed = {t for tools, _f in defs.values() for t in tools}
    with_tools = [f for f in cfs if f.tools]
    unclaimed = [f for f in with_tools if any(t.name not in claimed for t in f.tools)]
    units: List[Tuple[str, str, List[CodeFile], Optional[set]]] = []
    rest = cfs
    only: Optional[set] = None
    if len(defs) >= 2 or (defs and unclaimed):
        for dname, (tools, home) in defs.items():
            files = [home] + [f for f in with_tools if f is not home and any(t.name in tools for t in f.tools)]
            units.append((f"{project}#{dname}", dname, files, set(tools)))
        rest = unclaimed
        only = {t.name for f in unclaimed for t in f.tools if t.name not in claimed}
    # folders that hold tools
    folders: Dict[str, List[CodeFile]] = OrderedDict()
    for f in rest:
        if f.tools:
            folders.setdefault(_home_dir(project, f.path), []).append(f)
    merged = True
    while merged and len(folders) > 1:  # a folder whose modules another agent folder imports belongs to it
        merged = False
        for d in list(folders):
            importer = next((e for e in folders if e != d and any(_imports_folder(f, d) for f in folders[e] if d)), None)
            if importer is not None:
                folders[importer] += folders.pop(d)
                merged = True
                break
    if len(folders) <= 1 and not units:
        return [(project, name, cfs, None)]
    for d, files in folders.items():
        # agent files without tools (an entrypoint, routes, a guard) join the folder they are in
        files = files + [f for f in rest if not f.tools and f.agent_like and _within(_rel_dir(project, f.path), d)
                         and not any(_within(_rel_dir(project, f.path), e) and len(e) > len(d) for e in folders)]
        if len(folders) == 1 and not d:
            files += [f for f in rest if not f.tools and f not in files]
        path = f"{project}/{d}" if d else project
        pp = PurePosixPath(path)
        shown = (f"{pp.parent.name}/{pp.name}" if d else name) if len(folders) > 1 or units else name
        units.append((path, shown, files, only))
    return units


def build_code_agents(probe, files: List[CodeFile], roots: List[str], cfg: ConfigScan) -> List[Agent]:
    by_project: "OrderedDict[str, List[CodeFile]]" = OrderedDict()
    for cf in sorted(files, key=lambda f: f.path):
        by_project.setdefault(project_of(probe, cf.path, roots), []).append(cf)
    groups: List[Tuple[str, str, List[CodeFile], Optional[set]]] = []
    for project, cfs in by_project.items():
        groups += _units(project, cfs)
    agents: List[Agent] = []
    for unit, shown, cfs, only in groups:
        agentic = [f for f in cfs if f.agent_like]
        if not agentic:
            continue
        entry = (
            next((f for f in agentic if f.has_main), None)
            or next((f for f in agentic if f.routes), None)
            or max(agentic, key=lambda f: (len(f.tools), -len(f.path)))
        )
        project = unit.split("#", 1)[0]
        a = Agent(id=f"code:{unit}", display_name=shown, kind="code", entrypoint=entry.path, project=project)
        for f in cfs:
            for fw in f.frameworks:
                if fw not in a.framework and fw != "csl-core":
                    a.framework.append(fw)
            for m in f.model_ids:
                if m not in a.model_ids:
                    a.model_ids.append(m)
            for t in f.tools:
                if any(t.name == x.name for x in a.tools) or (only is not None and t.name not in only):
                    continue
                cls, why = classify(t, f.tool_calls.get(t.name))
                t.risk_class, t.risk_reason = cls, why
                a.tools.append(t)
            if f.prompt.present and (a.system_prompt.length or 0) < (f.prompt.length or 0):
                a.system_prompt = PromptInfo(True, f.prompt.length, f.prompt.sha256)
            for r in f.routes:
                a.triggers.append(Trigger(route_trigger_type(r.path, r.handler), r.path, f"{f.path}:{r.line}"))
            for name in f.env_names:
                kind = classify_credential(name)
                if kind and not any(c.name == name for c in a.access.credentials):
                    a.access.credentials.append(Credential(name, f.path, kind))
            for call, line, policy in f.guard_calls:
                a.guard.status = "wired"
                a.guard.mechanism = a.guard.mechanism or ("plugin" if call == "OpenClawGuard" else "wrapper")
                a.guard.evidence.append(Evidence("governance", f.path, line, f"{call}({policy or ''})"))
                if policy and policy not in a.guard.policy_ids:
                    a.guard.policy_ids.append(policy)
            a.evidence.append(Evidence("code", f.path, None, ", ".join(filter(None, [
                f"{len(f.tools)} tools" if f.tools else "", "entrypoint" if f is entry else "",
            ])) or None))
        pc = cfg.projects.get(project)
        for p in [pc] if pc else []:
            for c in p.credentials:
                if not any(x.name == c.name for x in a.access.credentials):
                    a.access.credentials.append(c)
            if p.crewai_agents and "crewai" not in a.framework:
                a.framework.append("crewai")
            a.evidence += p.evidence
        if a.triggers:
            a.kind = "service"
        agents.append(a)
    return agents


def build_assistant_agents(cfg: ConfigScan, home: str, window_days: int, now) -> List[Agent]:
    return [assistant_agent(rec, window_days, now) for rec in cfg.assistants]


def assistant_agent(rec: AssistantRecord, window_days: int, now) -> Agent:
    a = Agent(id=rec.id, display_name=rec.display_name, kind="assistant", project=rec.project,
              framework=[rec.product], state="configured")
    a.tools = builtin_tools(rec.product)
    seen_servers = set()
    for srv in rec.mcp_servers:
        if srv.name in seen_servers:
            continue
        seen_servers.add(srv.name)
        for t in mcp_tools(srv.name, srv.package):
            cls, why = classify(t)
            t.risk_class, t.risk_reason = cls, why
            t.evidence = [Evidence("config", srv.source, None, f"MCP server '{srv.name}'")]
            a.tools.append(t)
        for r in srv.fs_roots:
            if r not in a.access.fs_roots:
                a.access.fs_roots.append(r)
        for k in srv.env_keys:
            kind = classify_credential(k) or "other"
            a.access.credentials.append(Credential(k, srv.source, kind))
    a.access.credentials += [c for c in rec.credentials if not any(x.name == c.name for x in a.access.credentials)]
    a.access.permission_mode = rec.permission_mode
    for event, cmd in rec.hooks:
        if event == "PreToolUse" and "cslcore" in cmd:
            a.guard.status = "wired"
            a.guard.mechanism = "hook"
            a.guard.mode = "log" if "log" in cmd.split() or "--log" in cmd else "block"
            for tok in cmd.split():
                if tok.endswith(".csl") and tok not in a.guard.policy_ids:
                    a.guard.policy_ids.append(tok)
            a.guard.evidence.append(Evidence("config", "hooks.PreToolUse", None, cmd[:120]))
    if rec.sessions is not None:
        a.sessions = rec.sessions
        a.runs = stats_from_days(rec.session_days, now, window_days, "assistant session files", rec.last_session)
    a.evidence = list(rec.evidence)
    return a


def attach_triggers(agents: List[Agent], trig: TriggerScan, probe) -> List[Agent]:
    new: List[Agent] = []
    for tr in trig.records:
        paths = referenced_paths(tr.command)
        target = None
        for a in agents:
            if a.kind == "assistant":
                continue
            if any(p == a.entrypoint or (a.project and _within(p, a.project)) for p in paths):
                target = a
                break
        sched = humanize_cron(tr.schedule) if tr.schedule and tr.source.startswith(("crontab", "/etc/cron")) else tr.schedule
        trigger = Trigger("time" if tr.type == "time" else ("manual" if tr.type == "manual" else "boot"), sched, tr.source)
        if target is None:
            product = assistant_product(tr.command)
            if not (product or AGENT_CLI.search(tr.command) or " -p " in f" {tr.command} " and "claude" in tr.command):
                continue
            target = Agent(id=f"scheduled:{tr.unit or tr.source}", display_name=(tr.unit or tr.source).split(":")[-1].replace(".service", ""),
                           kind="scheduled", entrypoint=tr.command[:200], framework=[product] if product else [])
            new.append(target)
        target.triggers.append(trigger)
        target.evidence.append(Evidence("triggers", tr.source, None, tr.command[:160]))
        if tr.run_as:
            target.process_user = target.process_user or tr.run_as
            if tr.run_as == "root":
                target.access.elevated = True
        if tr.unit and tr.unit.endswith(".service"):
            target.evidence.append(Evidence("triggers", f"unit:{tr.unit}", None, None))
    return new


def attach_runtime(agents: List[Agent], rt: RuntimeScan, probe, home: str, cfg: ConfigScan,
                   window_days: int, now) -> Tuple[List[Agent], List[Listener]]:
    new: List[Agent] = []
    listeners: List[Listener] = []
    by_project = sorted([a for a in agents if a.project and a.kind != "assistant"], key=lambda a: -len(a.project or ""))
    for p in rt.processes:
        args = p.args
        addrs = rt.listening.get(p.pid, [])
        is_mcp = bool(MCP_SERVER.search(args))
        if addrs and (is_mcp or any(not is_loopback(x) for x in addrs)):
            listeners.append(Listener(p.pid, args[:200], addrs, is_mcp))
        target: Optional[Agent] = None
        for a in by_project:
            if (a.entrypoint and a.entrypoint in args) or (a.project and f"{a.project}/" in args):
                target = a
                break
        product = None if target else assistant_product(args)
        if target is None and product:
            cwd = process_cwd(probe, p.pid) if product in ("claude-code", "codex", "aider", "gemini-cli", "goose", "openclaw") else None
            cands = [a for a in agents if a.kind == "assistant" and a.framework[:1] == [product]]
            if cwd:
                exact = [a for a in cands if a.project and _within(cwd, a.project)]
                cands = sorted(exact, key=lambda a: -len(a.project or ""))[:1]
                if not cands and product == "claude-code":
                    import copy
                    base = copy.deepcopy(cfg.user_claude) if cfg.user_claude else None
                    rec = base or AssistantRecord(id="", display_name="", product="claude-code")
                    rec.id, rec.project = f"assistant:claude-code:{cwd}", cwd
                    rec.display_name = f"claude-code:{PurePosixPath(cwd).name or cwd}"
                    t = assistant_agent(rec, window_days, now)
                    new.append(t)
                    agents.append(t)
                    cands = [t]
            elif product == "claude-code":
                cands = []
            target = cands[0] if cands else None
            if target is None and product not in ("claude-code",):
                target = Agent(id=f"unmanaged:{product}:{p.pid}", display_name=product, kind="unmanaged", framework=[product])
                new.append(target)
                agents.append(target)
        if target is None and AGENT_CLI.search(args) and not is_mcp:
            script = next((x for x in args.split() if x.endswith(".py")), args.split()[0])
            sp = PurePosixPath(script)
            target = Agent(id=f"unmanaged:{p.pid}", display_name=f"{sp.parent.name}/{sp.name}" if sp.parent.name else sp.name, kind="unmanaged",
                           entrypoint=args[:160])
            new.append(target)
            agents.append(target)
        if target is None:
            continue
        target.pids.append(p.pid)
        target.state = "running"
        target.process_user = p.user
        if p.uid == 0:
            target.access.elevated = True
        if PERMISSION_SKIP.search(args):
            target.access.permission_mode = "bypass"
            target.evidence.append(Evidence("runtime", f"pid {p.pid}", None, "permission prompts disabled by flag"))
        for x in addrs:
            if x not in target.access.network_listen:
                target.access.network_listen.append(x)
        if not any(e.layer == "runtime" and e.path == f"pid {p.pid}" for e in target.evidence):
            target.evidence.append(Evidence("runtime", f"pid {p.pid}", None, f"user {p.user}"))
    for c in rt.containers:
        hay = f"{c.name} {c.image} {c.command}".lower()
        target = next((a for a in agents if a.project and PurePosixPath(a.project).name.lower() in hay), None)
        if target is None:
            continue
        if c.state == "running":
            target.state = "running"
        target.evidence.append(Evidence("runtime", f"docker:{c.name}", None, f"{c.image} ({c.state})"))
        if c.ports:
            for part in c.ports.split(","):
                part = part.strip()
                if "->" in part:
                    target.access.network_listen.append(part.split("->")[0])
    return new, listeners


def finalize_states(agents: List[Agent]) -> None:
    for a in agents:
        if a.state == "running":
            continue
        if any(t.type == "time" for t in a.triggers):
            a.state = "scheduled"
        elif a.kind == "assistant":
            a.state = "configured"
        elif a.kind == "service" or any(t.type in ("boot", "inbound_http", "messaging", "email") for t in a.triggers):
            a.state = "stopped"
        elif a.entrypoint:
            a.state = "stopped"
        else:
            a.state = "unknown"
        if a.kind == "code" and a.state == "scheduled":
            a.kind = "scheduled"


def attach_history(agents: List[Agent], probe, now, window_days: int, workspace_root: Optional[str]) -> None:
    from .layers.history import decision_log_runs

    for a in agents:
        if a.runs.count is not None:
            continue
        unit = next((e.path[5:] for e in a.evidence if e.path.startswith("unit:")), None)
        stats: Optional[RunStats] = None
        if unit:
            stats = journal_starts(probe, unit, now, window_days)
        if stats is None and any(t.source and t.source.startswith(("crontab", "/etc/cron")) for t in a.triggers) and a.entrypoint:
            stats = cron_log_runs(probe, a.entrypoint, now, window_days)
        if stats is None and workspace_root:
            safe = a.id.replace("/", "_").replace(":", "_")
            stats = decision_log_runs(probe, f"{workspace_root}/.csl/venom/decisions/{safe}.jsonl", now, window_days)
        if stats is not None:
            a.runs = stats
        else:
            a.runs = RunStats(count=None, window=f"{window_days}d", source=None)


def order_agents(agents: List[Agent]) -> List[Agent]:
    state_rank = {"running": 0, "scheduled": 1, "stopped": 2, "configured": 3, "unknown": 4}
    return sorted(agents, key=lambda a: (state_rank.get(a.state, 5), a.display_name.lower(), a.id))


def resolve(probe, roots, code_files, cfg, trig, rt, window_days, workspace_root) -> Resolution:
    home = probe.home()
    now = probe.now()
    agents = build_code_agents(probe, code_files, roots, cfg)
    agents += build_assistant_agents(cfg, home, window_days, now)
    if trig is not None:
        agents += attach_triggers(agents, trig, probe)
    listeners: List[Listener] = []
    if rt is not None:
        _new, listeners = attach_runtime(agents, rt, probe, home, cfg, window_days, now)
    finalize_states(agents)
    attach_history(agents, probe, now, window_days, workspace_root)
    _unique_names(agents)
    for a in agents:
        a.tools.sort(key=lambda t: t.name.lower())
    return Resolution(order_agents(agents), listeners)
