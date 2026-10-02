"""
L2 config layer.

Configured assistants, attached MCP servers, permission modes, hooks, skills and
credential names. Values of credentials are never kept: only the key name and the file.
Assistant session transcripts are counted from directory listings, never opened.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from ..model import Credential, Evidence

# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------

_CRED_RULES = [
    ("cloud_root", re.compile(r"^(AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|AWS_SESSION_TOKEN|GOOGLE_APPLICATION_CREDENTIALS|GCP_.*KEY|AZURE_(CLIENT_SECRET|.*KEY)|DIGITALOCEAN_TOKEN|CLOUDFLARE_API_(TOKEN|KEY)|HCLOUD_TOKEN)$")),
    ("payment", re.compile(r"(STRIPE|PAYPAL|BRAINTREE|ADYEN|PLAID|WALLET|MNEMONIC|PRIVATE_KEY|SEED_PHRASE)")),
    ("llm_key", re.compile(r"(OPENAI|ANTHROPIC|CLAUDE|GEMINI|GOOGLE_API_KEY|MISTRAL|COHERE|GROQ|DEEPSEEK|TOGETHER|OPENROUTER|HF_TOKEN|HUGGINGFACE|REPLICATE|PERPLEXITY|XAI).*(KEY|TOKEN)")),
    ("org_token", re.compile(r"^(GITHUB_TOKEN|GH_TOKEN|GITHUB_PAT|GITLAB_TOKEN|SLACK_(BOT_|APP_|USER_)?TOKEN|DISCORD_(BOT_)?TOKEN|TELEGRAM_BOT_TOKEN|NOTION_(API_)?KEY|LINEAR_API_KEY|JIRA_API_TOKEN|ATLASSIAN_.*TOKEN|NPM_TOKEN|PYPI_TOKEN|TWILIO_AUTH_TOKEN|SENDGRID_API_KEY|VERCEL_TOKEN)$")),
    ("database_url", re.compile(r"(DATABASE_URL|POSTGRES|PG_(PASSWORD|URL)|MYSQL|MONGO|REDIS_URL|SUPABASE_(SERVICE|KEY)|DB_(PASSWORD|URL))")),
]
_SECRETISH = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_URL|DSN|PRIVATE|AUTH)", re.I)


def classify_credential(name: str) -> Optional[str]:
    up = name.upper()
    for kind, rx in _CRED_RULES:
        if rx.search(up):
            return kind
    return "other" if _SECRETISH.search(up) else None


def env_file_keys(text: str) -> List[str]:
    keys = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if m and m.group(1) not in keys:
            keys.append(m.group(1))
    return keys


def credentials_from_keys(keys: List[str], file: str) -> List[Credential]:
    out = []
    for k in keys:
        kind = classify_credential(k)
        if kind:
            out.append(Credential(name=k, file=file, kind=kind))
    return out


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass
class McpServer:
    name: str
    command: Optional[str]
    args: List[str]
    env_keys: List[str]
    url: Optional[str]
    source: str
    fs_roots: List[str] = field(default_factory=list)
    package: Optional[str] = None
    # env values are kept in memory only to start the server under --probe; never rendered or stored
    env_values: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class AssistantRecord:
    id: str
    display_name: str
    product: str  # claude-code | claude-desktop | cursor | vscode | windsurf
    project: Optional[str] = None
    permission_mode: Optional[str] = None
    allow_rules: List[str] = field(default_factory=list)
    deny_rules: List[str] = field(default_factory=list)
    hooks: List[Tuple[str, str]] = field(default_factory=list)  # (event, command)
    mcp_servers: List[McpServer] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    sessions: Optional[int] = None
    session_days: Dict[str, int] = field(default_factory=dict)
    last_session: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    credentials: List[Credential] = field(default_factory=list)


@dataclass
class ProjectConfig:
    path: str
    credentials: List[Credential] = field(default_factory=list)
    crewai_agents: List[str] = field(default_factory=list)
    compose_services: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class ConfigScan:
    assistants: List[AssistantRecord] = field(default_factory=list)
    projects: Dict[str, ProjectConfig] = field(default_factory=dict)
    files_read: int = 0
    older_projects: int = 0
    user_claude: Optional[AssistantRecord] = None  # user-level Claude Code settings, applied to new projects


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------

def _load_json(probe, path: str) -> Optional[Dict[str, Any]]:
    text = probe.read_text(path, limit=5_000_000)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # VS Code style JSON with comments
        try:
            data = json.loads(re.sub(r"(?m)^\s*//.*$", "", text))
        except (json.JSONDecodeError, ValueError):
            return None
    return data if isinstance(data, dict) else None


_FS_PACKAGES = ("server-filesystem", "mcp-server-filesystem", "filesystem")


def parse_mcp_servers(block: Any, source: str, home: str) -> List[McpServer]:
    out: List[McpServer] = []
    if not isinstance(block, dict):
        return out
    for name, spec in sorted(block.items()):
        if not isinstance(spec, dict):
            continue
        args = [str(a) for a in spec.get("args", []) if isinstance(a, (str, int))]
        env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
        command = spec.get("command") if isinstance(spec.get("command"), str) else None
        pkg = next((a for a in args if a.startswith("@") or a.startswith("mcp-") or "server-" in a or a.endswith("-mcp")), None)
        if pkg is None and command and command not in ("npx", "uvx", "node", "python", "python3", "docker", "bunx", "pipx"):
            pkg = PurePosixPath(command).name
        roots: List[str] = []
        joined = " ".join([command or ""] + args)
        if any(p in joined for p in _FS_PACKAGES):
            for a in args:
                if a.startswith("/") or a.startswith("~") or a == ".":
                    roots.append(home if a in ("~", "~/") else a.replace("~", home, 1) if a.startswith("~") else a)
        out.append(McpServer(
            name=name, command=command, args=args, env_keys=sorted(env.keys()),
            url=spec.get("url") if isinstance(spec.get("url"), str) else None,
            source=source, fs_roots=roots, package=pkg,
            env_values={str(k): str(v) for k, v in env.items()},
        ))
    return out


def _hooks(settings: Dict[str, Any]) -> List[Tuple[str, str]]:
    out = []
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return out
    for event, entries in hooks.items():
        for entry in entries if isinstance(entries, list) else []:
            for h in entry.get("hooks", []) if isinstance(entry, dict) else []:
                if isinstance(h, dict) and isinstance(h.get("command"), str):
                    out.append((str(event), h["command"]))
    return out


def _permission_mode(settings: Dict[str, Any]) -> Optional[str]:
    perms = settings.get("permissions") if isinstance(settings.get("permissions"), dict) else {}
    mode = perms.get("defaultMode") or settings.get("defaultMode")
    if mode == "bypassPermissions":
        return "bypass"
    return mode if isinstance(mode, str) else None


def _merge_settings(rec: AssistantRecord, settings: Dict[str, Any], path: str, home: str) -> None:
    mode = _permission_mode(settings)
    if mode:
        rec.permission_mode = mode
    perms = settings.get("permissions") if isinstance(settings.get("permissions"), dict) else {}
    rec.allow_rules += [r for r in perms.get("allow", []) if isinstance(r, str)]
    rec.deny_rules += [r for r in perms.get("deny", []) if isinstance(r, str)]
    rec.hooks += _hooks(settings)
    rec.mcp_servers += parse_mcp_servers(settings.get("mcpServers"), path, home)
    env = settings.get("env")
    if isinstance(env, dict):
        rec.credentials += credentials_from_keys(sorted(env.keys()), path)
    rec.evidence.append(Evidence("config", path, None, "settings"))


def decode_project_dir(encoded: str) -> str:
    """Claude Code encodes project paths by replacing '/' (and '.') with '-'."""
    return "/" + encoded.lstrip("-").replace("-", "/")


def _resolve_project(probe, encoded: str) -> Optional[str]:
    """Best effort: walk the encoded name, preferring real folders, so dashes in names survive."""
    parts = encoded.lstrip("-").split("-")
    cur = ""
    i = 0
    while i < len(parts):
        matched = False
        for j in range(len(parts), i, -1):
            for sep in ("-", ".", "_", " "):
                cand = f"{cur}/{sep.join(parts[i:j])}"
                if probe.is_dir(cand):
                    cur, i, matched = cand, j, True
                    break
            if matched:
                break
        if not matched:
            return None
    return cur or None


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def _app_support(probe, *parts: str) -> List[str]:
    home = probe.home()
    if probe.os_name() == "macos":
        return [str(PurePosixPath(home, "Library", "Application Support", *parts))]
    return [str(PurePosixPath(home, ".config", *parts))]


def scan_config(probe, roots: List[str], host_level: bool, window_days: int = 7, recent_days: int = 30) -> ConfigScan:
    home = probe.home()
    now = probe.now()
    scan = ConfigScan()
    claude_projects: Dict[str, AssistantRecord] = {}

    def cc_project(project: str) -> AssistantRecord:
        if project not in claude_projects:
            claude_projects[project] = AssistantRecord(
                id=f"assistant:claude-code:{project}", display_name=f"claude-code:{PurePosixPath(project).name or project}",
                product="claude-code", project=project,
            )
        return claude_projects[project]

    user_level: Optional[AssistantRecord] = None
    if host_level:
        user_level = AssistantRecord(id="assistant:claude-code:user", display_name="claude-code (user)", product="claude-code")
        for name in ("settings.json", "settings.local.json"):
            p = f"{home}/.claude/{name}"
            data = _load_json(probe, p)
            if data is not None:
                scan.files_read += 1
                _merge_settings(user_level, data, p, home)
        # ~/.claude.json: user-scope and per-project MCP servers
        cj = _load_json(probe, f"{home}/.claude.json")
        if cj is not None:
            scan.files_read += 1
            user_level.mcp_servers += parse_mcp_servers(cj.get("mcpServers"), f"{home}/.claude.json", home)
            projects = cj.get("projects") if isinstance(cj.get("projects"), dict) else {}
            for proj, pdata in projects.items():
                if isinstance(pdata, dict) and pdata.get("mcpServers"):
                    rec = cc_project(proj)
                    rec.mcp_servers += parse_mcp_servers(pdata.get("mcpServers"), f"{home}/.claude.json", home)
        skills_dir = f"{home}/.claude/skills"
        for s in probe.list_dir(skills_dir) if probe.is_dir(skills_dir) else []:
            if probe.exists(f"{skills_dir}/{s}/SKILL.md"):
                user_level.skills.append(s)
        # sessions: count + mtime only, from directory listings
        pdir = f"{home}/.claude/projects"
        for enc in sorted(probe.list_dir(pdir)) if probe.is_dir(pdir) else []:
            sdir = f"{pdir}/{enc}"
            if not probe.is_dir(sdir):
                continue
            count = 0
            days: Dict[str, int] = {}
            newest = 0.0
            for f in probe.list_dir(sdir):
                if not f.endswith(".jsonl"):
                    continue
                st = probe.stat(f"{sdir}/{f}")
                if st is None:
                    continue
                count += 1
                newest = max(newest, st[1])
                age = (now.timestamp() - st[1]) / 86400
                if 0 <= age < window_days:
                    day = datetime.fromtimestamp(st[1], timezone.utc).strftime("%Y-%m-%d")
                    days[day] = days.get(day, 0) + 1
            if not count:
                continue
            if (now.timestamp() - newest) / 86400 > recent_days:
                scan.older_projects += 1
                continue
            project = _resolve_project(probe, enc) or decode_project_dir(enc)
            rec = cc_project(project)
            rec.sessions = count
            rec.session_days = days
            rec.last_session = datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec.evidence.append(Evidence("history", sdir, None, f"{count} session files (not opened)"))

        # other assistants
        for path in _app_support(probe, "Claude", "claude_desktop_config.json"):
            data = _load_json(probe, path)
            if data is not None:
                scan.files_read += 1
                rec = AssistantRecord(id="assistant:claude-desktop", display_name="claude-desktop", product="claude-desktop")
                rec.mcp_servers = parse_mcp_servers(data.get("mcpServers"), path, home)
                rec.evidence.append(Evidence("config", path, None, "desktop config"))
                scan.assistants.append(rec)
        for product, paths in (
            ("cursor", [f"{home}/.cursor/mcp.json"]),
            ("windsurf", [f"{home}/.codeium/windsurf/mcp_config.json"]),
            ("vscode", _app_support(probe, "Code", "User", "mcp.json")),
        ):
            for path in paths:
                data = _load_json(probe, path)
                if data is None:
                    continue
                scan.files_read += 1
                rec = AssistantRecord(id=f"assistant:{product}", display_name=product, product=product)
                rec.mcp_servers = parse_mcp_servers(data.get("mcpServers") or data.get("servers"), path, home)
                rec.evidence.append(Evidence("config", path, None, "MCP config"))
                scan.assistants.append(rec)

    # project-level configuration inside the scan roots
    for root in roots:
        _scan_project_dirs(probe, root, scan, cc_project, home)

    scan.user_claude = user_level
    if user_level is not None and (user_level.evidence or user_level.mcp_servers):
        scan.assistants.insert(0, user_level)
    for proj in sorted(claude_projects):
        rec = claude_projects[proj]
        if user_level is not None:
            # user-level settings apply to every project unless the project overrides them
            if rec.permission_mode is None:
                rec.permission_mode = user_level.permission_mode
            rec.hooks = user_level.hooks + rec.hooks
            rec.mcp_servers = user_level.mcp_servers + [m for m in rec.mcp_servers if m.name not in {u.name for u in user_level.mcp_servers}]
            rec.skills = list(user_level.skills)
        scan.assistants.append(rec)
    if user_level is not None and claude_projects and user_level in scan.assistants:
        # the user-level record is folded into project records when projects exist
        scan.assistants.remove(user_level)
    return scan


def _scan_project_dirs(probe, root: str, scan: ConfigScan, cc_project, home: str, depth: int = 0) -> None:
    if depth > 3 or not probe.is_dir(root):
        return
    try:
        names = probe.list_dir(root)
    except PermissionError:
        return
    pc: Optional[ProjectConfig] = None

    def project() -> ProjectConfig:
        nonlocal pc
        if pc is None:
            pc = scan.projects.setdefault(root, ProjectConfig(path=root))
        return pc

    if ".claude" in names and probe.is_dir(f"{root}/.claude") and root != home:
        for name in ("settings.json", "settings.local.json"):
            p = f"{root}/.claude/{name}"
            data = _load_json(probe, p)
            if data is not None:
                scan.files_read += 1
                _merge_settings(cc_project(root), data, p, home)
    if ".mcp.json" in names:
        data = _load_json(probe, f"{root}/.mcp.json")
        if data is not None:
            scan.files_read += 1
            cc_project(root).mcp_servers += parse_mcp_servers(data.get("mcpServers"), f"{root}/.mcp.json", home)
    for sub, product in ((".cursor", "cursor"), (".vscode", "vscode")):
        p = f"{root}/{sub}/mcp.json"
        if sub in names and probe.exists(p):
            data = _load_json(probe, p)
            if data is not None:
                scan.files_read += 1
                rec = AssistantRecord(id=f"assistant:{product}:{root}", display_name=f"{product}:{PurePosixPath(root).name}", product=product, project=root)
                rec.mcp_servers = parse_mcp_servers(data.get("mcpServers") or data.get("servers"), p, home)
                rec.evidence.append(Evidence("config", p, None, "project MCP config"))
                scan.assistants.append(rec)
    for envname in (".env", ".env.local", ".env.production"):
        if envname in names:
            text = probe.read_text(f"{root}/{envname}")
            if text is not None:
                scan.files_read += 1
                creds = credentials_from_keys(env_file_keys(text), f"{root}/{envname}")
                if creds:
                    project().credentials += creds
                    project().evidence.append(Evidence("config", f"{root}/{envname}", None, f"{len(creds)} credential names"))
    for compose in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        if compose in names:
            text = probe.read_text(f"{root}/{compose}")
            if text is not None:
                scan.files_read += 1
                services, keys = _parse_compose(text)
                project().compose_services += services
                project().credentials += credentials_from_keys(keys, f"{root}/{compose}")
    for crew in ("config/agents.yaml", "src/config/agents.yaml"):
        if probe.exists(f"{root}/{crew}"):
            text = probe.read_text(f"{root}/{crew}")
            if text is not None:
                project().crewai_agents += re.findall(r"(?m)^([A-Za-z_][\w\-]*):\s*$", text)
                project().evidence.append(Evidence("config", f"{root}/{crew}", None, "crewai agents"))
    for n in sorted(names):
        if n.startswith(".") or n in ("node_modules", "venv", ".venv", "__pycache__", "dist", "build", "site-packages", "Library"):
            continue
        p = f"{root}/{n}"
        if probe.is_dir(p):
            _scan_project_dirs(probe, p, scan, cc_project, home, depth + 1)


def _parse_compose(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Minimal YAML reading for compose files: service names, images, commands, env key names."""
    services: List[Dict[str, Any]] = []
    keys: List[str] = []
    in_services = False
    cur: Optional[Dict[str, Any]] = None
    in_env = False
    for raw in text.splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 0:
            in_services = s == "services:"
            cur = None
            continue
        if not in_services:
            continue
        if indent == 2 and s.endswith(":"):
            cur = {"name": s[:-1], "image": None, "command": None, "ports": []}
            services.append(cur)
            in_env = False
            continue
        if cur is None:
            continue
        if indent == 4:
            in_env = s.startswith("environment")
            if s.startswith("image:"):
                cur["image"] = s.split(":", 1)[1].strip()
            elif s.startswith("command:"):
                cur["command"] = s.split(":", 1)[1].strip()
            continue
        if in_env and indent >= 6:
            m = re.match(r"^-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]", s)
            if m:
                keys.append(m.group(1))
    return services, keys
