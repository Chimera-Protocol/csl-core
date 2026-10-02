"""The same sorter, with the derived values computed by the hardened classifiers."""
from chimera_core.mapping import command_allowed, destination_allowed, in_scope, to_enum

TOOLS = ["Bash", "Write", "Edit", "Read", "WebFetch"]
ROOTS = ["/srv/app"]
COMMANDS = ["git status", "git log *", "ls"]
HOSTS = ["api.example.com"]


def classify(tool_name, args):
    return {
        "tool": to_enum(tool_name, TOOLS, name="tool"),
        "path_ok": in_scope(args.get("file_path"), ROOTS),
        "cmd_ok": command_allowed(args.get("command"), COMMANDS),
        "dest_ok": destination_allowed(args.get("url"), HOSTS),
    }
