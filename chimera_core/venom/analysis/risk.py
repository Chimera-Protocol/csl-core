"""
Tool risk classification.

Order of evidence: known MCP server catalog, calls made inside the tool body, tool name
and description keywords, parameter names. Anything unmatched is UNCLASSIFIED, which
counts as sensitive.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..model import Tool, ToolParam

# Known MCP servers: package or server-name fragment -> (default class, {tool: class})
MCP_CATALOG: Dict[str, Tuple[str, Dict[str, str]]] = {
    "server-filesystem": ("WRITE", {
        "read_file": "READ", "read_text_file": "READ", "read_multiple_files": "READ", "list_directory": "READ",
        "directory_tree": "READ", "search_files": "READ", "get_file_info": "READ", "list_allowed_directories": "READ",
        "write_file": "WRITE", "edit_file": "WRITE", "create_directory": "WRITE", "move_file": "WRITE",
    }),
    "server-github": ("EXTERNAL", {
        "search_repositories": "READ", "get_file_contents": "READ", "list_issues": "READ",
        "create_issue": "EXTERNAL", "create_pull_request": "EXTERNAL", "push_files": "WRITE",
        "create_or_update_file": "WRITE", "merge_pull_request": "WRITE",
    }),
    "github-mcp-server": ("EXTERNAL", {}),
    "server-slack": ("EXTERNAL", {"slack_list_channels": "READ", "slack_post_message": "EXTERNAL", "slack_reply_to_thread": "EXTERNAL"}),
    "server-postgres": ("READ", {"query": "READ"}),
    "server-sqlite": ("WRITE", {"read_query": "READ", "write_query": "WRITE", "create_table": "WRITE"}),
    "server-fetch": ("EXTERNAL", {"fetch": "EXTERNAL"}),
    "mcp-server-fetch": ("EXTERNAL", {"fetch": "EXTERNAL"}),
    "server-puppeteer": ("EXTERNAL", {}),
    "playwright": ("EXTERNAL", {}),
    "server-brave-search": ("READ", {}),
    "server-memory": ("WRITE", {}),
    "server-git": ("WRITE", {"git_status": "READ", "git_diff": "READ", "git_log": "READ", "git_commit": "WRITE", "git_reset": "DESTRUCTIVE"}),
    "desktop-commander": ("EXEC", {}),
    "server-shell": ("EXEC", {}),
    "mcp-shell": ("EXEC", {}),
    "stripe": ("SPEND", {}),
    "server-aws": ("IDENTITY", {}),
    "aws-mcp": ("IDENTITY", {}),
    "csl-core-mcp": ("READ", {
        "verify_policy": "READ", "simulate_policy": "READ", "explain_policy": "READ", "scaffold_policy": "READ",
        "tla_verify": "READ", "universe_info": "READ",
    }),
    "chrome": ("EXTERNAL", {}),
    "notion": ("EXTERNAL", {}),
    "linear": ("EXTERNAL", {}),
    "sentry": ("READ", {}),
    "context7": ("READ", {}),
}

# Built-in tools of known assistants.
# Built-in tools of known assistants: (name, class, parameter names)
ASSISTANT_BUILTINS: Dict[str, List[Tuple[str, str, Tuple[str, ...]]]] = {
    "claude-code": [
        ("Bash", "EXEC", ("command",)), ("Write", "WRITE", ("file_path", "content")),
        ("Edit", "WRITE", ("file_path", "old_string", "new_string")), ("NotebookEdit", "WRITE", ("notebook_path",)),
        ("Read", "READ", ("file_path",)), ("Glob", "READ", ("pattern",)), ("Grep", "READ", ("pattern",)),
        ("WebFetch", "EXTERNAL", ("url", "prompt")), ("WebSearch", "READ", ("query",)),
    ],
    "cursor": [("terminal", "EXEC", ("command",)), ("edit_file", "WRITE", ("target_file",)), ("read_file", "READ", ("target_file",))],
    "windsurf": [("run_command", "EXEC", ("command",)), ("write_to_file", "WRITE", ("path",)), ("view_file", "READ", ("path",))],
    "vscode": [("runInTerminal", "EXEC", ("command",)), ("editFiles", "WRITE", ("path",)), ("readFile", "READ", ("path",))],
    "claude-desktop": [],
}

# qualified call names inside a tool body -> class. Only unambiguous library calls count:
# `dict.update` or `list.remove` must never make a tool look state-changing.
_BODY_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(shutil\.rmtree|os\.(remove|unlink|rmdir|removedirs))$|\.(rmtree|drop_table|delete_many|delete_one|drop_collection|truncate_table)$"), "DESTRUCTIVE"),
    (re.compile(r"^(subprocess\.\w+|os\.(system|popen|exec\w*|spawn\w*)|pty\.spawn|eval|exec|asyncio\.create_subprocess_\w+|pexpect\.\w+)$"), "EXEC"),
    (re.compile(r"^(stripe\.\w+(\.\w+)*|paypalrestsdk\.\w+)$|\.(send_transaction|sendTransaction|create_payment|create_payout|transfer_funds)$"), "SPEND"),
    (re.compile(r"\.(create_user|attach_role_policy|put_role_policy|attach_user_policy|create_access_key|add_role_member)$"), "IDENTITY"),
    (re.compile(r"^(requests|httpx|aiohttp\.ClientSession)\.(post|put|patch|delete)$|^smtplib\.\w+|\.(send_message|sendmail|send_mail|chat_postMessage|create_tweet)$"), "EXTERNAL"),
    (re.compile(r"\.(write_text|write_bytes)$|^(os\.makedirs|os\.mkdir|shutil\.(copy\w*|move))$"), "WRITE"),
]

_NAME_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(delete|remove|drop|destroy|purge|wipe|truncate|erase|terminate|revoke|reset|shutdown|kill|rm_|_rm$)", re.I), "DESTRUCTIVE"),
    (re.compile(r"(shell|bash|exec|command|terminal|run_code|python_repl|execute|eval|subprocess|script|sandbox)", re.I), "EXEC"),
    (re.compile(r"(transfer|payment|pay_|_pay$|^pay|charge|refund|purchase|buy|sell|trade|swap|withdraw|deposit|payout|wire|spend|provision|order|bid|mint|credit(?!_?(check|score|report|status|info|history|limit))|reimburs|rebate|compensat|top_?up|gift_card|voucher)", re.I), "SPEND"),
    (re.compile(r"(credential|password|secret|token|role|permission|grant|iam|access_key|api_key|user_admin|invite|auth)", re.I), "IDENTITY"),
    (re.compile(r"(send|email|mail|post|publish|tweet|slack|message|notify|sms|webhook|upload|share|http_request|fetch_url|browse|comment|reply|dm_)", re.I), "EXTERNAL"),
    (re.compile(r"(write|create|update|(^|_)edit|insert|save|set_|put_|modify|append|rename|move|commit|deploy|schedule|book|add_|patch)", re.I), "WRITE"),
    (re.compile(r"(read|get|list|search|query|find|lookup|fetch|view|show|describe|count|check|status|inspect|summar|analy|calculat|convert|parse|validate|explain|verify|simulate|info|weather|time$|echo|generate|scaffold|format|render|draft)", re.I), "READ"),
]

_DESC_WORDS: List[Tuple[str, List[str]]] = [
    ("DESTRUCTIVE", ["delete", "deletes", "remove", "removes", "drop", "destroy", "purge", "wipe", "erase", "terminate"]),
    ("EXEC", ["shell", "bash", "execute", "executes", "command", "commands", "terminal", "subprocess"]),
    ("SPEND", ["payment", "pay", "transfer", "charge", "purchase", "buy", "sell", "trade", "refund"]),
    ("IDENTITY", ["credential", "credentials", "password", "permission", "permissions", "role", "roles"]),
    ("EXTERNAL", ["send", "sends", "email", "post", "posts", "publish", "tweet", "message", "notify", "upload", "sms"]),
    ("WRITE", ["write", "writes", "create", "creates", "update", "updates", "insert", "save", "edit", "modify"]),
    ("READ", ["read", "reads", "get", "gets", "list", "lists", "search", "query", "fetch", "look", "return", "returns", "check", "summary", "analyze", "explain", "verify", "simulate", "generate", "scaffold", "format"]),
]


def _desc_class(desc: str) -> Optional[Tuple[str, str]]:
    words = set(re.findall(r"[a-z]+", desc.lower()))
    for cls, vocab in _DESC_WORDS:
        hit = next((w for w in vocab if w in words), None)
        if hit:
            return cls, f"description mentions '{hit}'"
    return None


# Tool names are usually verb_object: the leading verb decides before any keyword search.
_LEADING_VERBS: Dict[str, str] = {}
for _cls, _verbs in (
    ("READ", "get list read search query fetch find lookup view show describe count check inspect explain verify "
             "simulate summarize analyze calculate convert parse validate scaffold generate format render peek stat"),
    ("WRITE", "write create update edit insert save set put modify append rename move commit deploy add patch store mkdir propose"),
    ("EXTERNAL", "send post publish tweet notify email message share reply upload broadcast"),
    ("DESTRUCTIVE", "delete remove drop destroy purge wipe truncate erase terminate revoke reset kill rm"),
    ("EXEC", "run exec execute eval shell bash spawn invoke"),
    ("SPEND", "pay transfer charge refund purchase buy sell trade swap withdraw deposit payout wire spend bid mint"),
):
    for _v in _verbs.split():
        _LEADING_VERBS[_v] = _cls


def _leading_verb(name: str) -> Optional[Tuple[str, str]]:
    words = re.findall(r"[a-z]+|[A-Z][a-z]*", name)
    if not words:
        return None
    verb = words[0].lower()
    cls = _LEADING_VERBS.get(verb)
    return (cls, f"verb '{verb}'") if cls else None


_PARAM_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(command|cmd|script|code|shell)$", re.I), "EXEC"),
    (re.compile(r"(amount|price|wallet|iban|account_number|card)", re.I), "SPEND"),
    (re.compile(r"^(to|recipient|recipients|email|channel|url|webhook|phone)$", re.I), "EXTERNAL"),
    (re.compile(r"^(content|body|data|text)$", re.I), "WRITE"),
]

_RANK = {c: i for i, c in enumerate(["READ", "WRITE", "EXTERNAL", "IDENTITY", "EXEC", "SPEND", "DESTRUCTIVE"])}


def catalog_for(package_or_name: Optional[str]) -> Optional[Tuple[str, str, Dict[str, str]]]:
    if not package_or_name:
        return None
    low = package_or_name.lower()
    for key in sorted(MCP_CATALOG, key=len, reverse=True):
        if key in low:
            default, tools = MCP_CATALOG[key]
            return key, default, tools
    return None


def classify(tool: Tool, body_calls: Optional[List[str]] = None) -> Tuple[str, str]:
    """Return (risk_class, reason)."""
    if tool.mcp_server:
        cat = catalog_for(tool.mcp_server)
        if cat:
            key, default, tools = cat
            if tool.name in tools:
                return tools[tool.name], f"catalog {key}"
            if tool.name.endswith("/*"):
                return default, f"catalog {key} (server default)"
    body: Optional[Tuple[str, str]] = None
    for call in body_calls or []:
        for rx, cls in _BODY_RULES:
            if rx.search(call) and (body is None or _RANK[cls] > _RANK[body[0]]):
                body = (cls, f"calls {call}")
    verb = _leading_verb(tool.name)
    named: Optional[Tuple[str, str]] = None
    for rx, cls in _NAME_RULES:
        if rx.search(tool.name):
            named = (cls, f"name '{tool.name}'")
            break
    if verb is not None:
        # a reading verb is authoritative (reading moves nothing); otherwise the riskier of verb and keywords
        if verb[0] == "READ" or named is None or _RANK[verb[0]] >= _RANK[named[0]]:
            named = verb
    if named is None and tool.description:
        named = _desc_class(tool.description)
    best = named
    if body and (best is None or _RANK[body[0]] > _RANK[best[0]]):
        best = body
    for p in tool.params:
        for rx, cls in _PARAM_HINTS:
            if not rx.search(p.name):
                continue
            # a parameter can decide an unmatched tool, or lift a READ tool that takes a command or an amount;
            # not one whose name starts with a reading verb (check_balance(wallet) reads a balance)
            read_by_verb = verb is not None and verb[0] == "READ" and best is not None and best[0] == "READ"
            if best is None or (best[0] == "READ" and cls in ("EXEC", "SPEND") and not read_by_verb):
                best = (cls, f"parameter '{p.name}'")
    return best or ("UNCLASSIFIED", "no evidence; counted as sensitive")


def mcp_tools(server_name: str, package: Optional[str]) -> List[Tool]:
    """Tools an attached MCP server exposes: catalog entries, else one server-wide entry."""
    cat = catalog_for(package) or catalog_for(server_name)
    label = server_name
    if cat and cat[2]:
        return [Tool(name=t, source="mcp_server", mcp_server=package or server_name, params=[]) for t in cat[2]]
    return [Tool(name=f"{label}/*", source="mcp_server", mcp_server=package or server_name,
                 description=f"all tools of MCP server '{label}' (not enumerated without --probe)")]


def builtin_tools(product: str) -> List[Tool]:
    return [Tool(name=n, source="builtin", risk_class=c, risk_reason=f"{product} built-in",
                 params=[ToolParam(p, "string") for p in ps]) for n, c, ps in ASSISTANT_BUILTINS.get(product, [])]


def param_type_label(p: ToolParam) -> str:
    return p.type or "any"
