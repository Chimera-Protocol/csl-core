"""A hand-written classifier with the usual mistakes (startswith, first word, substring)."""

SAFE = ("git", "ls", "cat")


def classify(tool_name, args):
    ctx = {"tool": tool_name}
    path = str(args.get("file_path", ""))
    ctx["path_ok"] = "YES" if path.startswith("/srv/app") else "NO"
    cmd = str(args.get("command", ""))
    ctx["cmd_ok"] = "YES" if cmd.split(" ")[0] in SAFE else "NO"
    url = str(args.get("url", ""))
    ctx["dest_ok"] = "YES" if "api.example.com" in url else "NO"
    return ctx
