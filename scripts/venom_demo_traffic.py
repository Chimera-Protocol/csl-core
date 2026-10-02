#!/usr/bin/env python3
"""
Demo traffic for `cslcore watch`: sends a steady stream of realistic tool calls through the
guards that `cslcore setup` created in a workspace, so the live panel has something to show.

    python scripts/venom_demo_traffic.py --workspace /tmp/venom-demo

Every call goes through the real generated mapping and policy (venom_guard), in the mode the
panel sets, so switching modes or disabling an agent in `cslcore watch` shows up at once.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

CALLS = {
    "membership-bot": [("check_balance", {}), ("transfer_funds", {"amount": 40, "to_wallet": "w-17"}),
                       ("transfer_funds", {"amount": 750, "to_wallet": "w-03"}), ("transfer_funds", {"amount": 25})],
    "ingest-worker": [("http_get", {"url": "https://partner.example/feed"}), ("run_command", {"command": "ls /data/in"}),
                      ("write_file", {"path": "/data/out/batch.json", "content": "..."})],
    "publisher": [("fetch_feed", {"limit": 10}), ("post_to_page", {"text": "Weekly update", "visibility": "public"})],
    "claude-code-ops": [("Read", {"file_path": "/srv/ops/README.md"}), ("Bash", {"command": "git status"}),
                        ("Edit", {"file_path": "/srv/ops/deploy.py"}), ("WebFetch", {"url": "https://docs.example"})],
    "claude-code-sandbox": [("Bash", {"command": "pytest -q"}), ("Write", {"file_path": "/srv/sandbox/x.py", "content": "x"})],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--rate", type=float, default=4.0, help="calls per second (default 4)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (default: run until Ctrl-C)")
    ap.add_argument("--count", type=int, default=0, help="send exactly this many calls without pausing, then stop")
    ap.add_argument("--seed", type=int, default=None, help="random seed, for the same traffic every run")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    from chimera_core.venom.observe import venom_guard

    ws = Path(args.workspace).resolve()
    state = json.loads((ws / ".csl/venom/state.json").read_text())
    guards = {}
    for st in (state.get("setup") or {}).get("agents", {}).values():
        key = st.get("key")
        if key in CALLS and st.get("mapping") and (ws / "policies" / f"{key}.csl").exists():
            guards[key] = venom_guard(key, policy=f"policies/{key}.csl", mapping=st["mapping"], workspace=str(ws))
    if not guards:
        print("no wired demo agents found; run: cslcore setup --root tests/venom/fixtures/host_ops --workspace", ws)
        return 2
    print(f"sending calls for {', '.join(sorted(guards))} · open another terminal: cslcore watch --workspace {ws}")
    t_end = time.monotonic() + args.seconds if args.seconds else None
    n = 0
    try:
        while (n < args.count) if args.count else (t_end is None or time.monotonic() < t_end):
            key = random.choice(list(guards))
            tool, call_args = random.choice(CALLS[key])
            guards[key].verify(tool, dict(call_args), {"approval": random.choice(["YES", "NO", "NO"])})
            n += 1
            if not args.count:
                time.sleep(1 / max(0.1, args.rate))
    except KeyboardInterrupt:
        pass
    print(f"{n} calls sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
