"""
Argument wiring for the 0.6 commands: setup, venom, studio, policy, map, exempt, mode, hook, watch.

This module only declares arguments; the Venom package (chimera_core.venom) is imported
inside the handlers, so `cslcore verify` and `import chimera_core` never load it.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Callable


def _lazy(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        module, _, func = name.rpartition(".")
        try:
            return getattr(importlib.import_module(module), func)(args)
        except SystemExit as e:
            if isinstance(e.code, str):  # a usage problem explained in words: show it, exit 2
                from rich.console import Console
                from rich.text import Text
                Console(highlight=False).print(Text.assemble(("cslcore: ", "bold red"), (e.code, "")))
                return 2
            raise
    handler.__name__ = name.rsplit(".", 1)[-1]
    return handler


def _shared(p: argparse.ArgumentParser, scan: bool = True) -> None:
    g = p.add_argument_group("shared options")
    if scan:
        g.add_argument("--root", metavar="PATH", help="scan this folder only (default: this host)")
        g.add_argument("--since", metavar="WINDOW", default="7d", help="run history window, e.g. 7d, 30d (default: 7d)")
        g.add_argument("--probe", action="store_true",
                       help="ask configured MCP servers for their real tool lists (starts stdio servers; asks first)")
    g.add_argument("--workspace", metavar="PATH", help="Venom workspace folder (default: current folder)")
    g.add_argument("--no-color", action="store_true", help="plain output (also honours NO_COLOR)")
    g.add_argument("--no-anim", action="store_true", help="no discovery animation (also CSL_NO_ANIM, CI)")
    g.add_argument("--plan-only", action="store_true", help="show what would be written, write nothing")


SETUP_HELP = "Guided flow: discover agents, write and verify policies, map and wire them"
VENOM_DESCRIPTION = (
    "Venom discovers the AI agents on this host (read-only): what exists, what runs, what\n"
    "each agent can reach and whether a guard is in the call path.\n\n"
    "Next steps: cslcore setup (guided), cslcore policy, cslcore map, cslcore watch."
)


def _setup_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--yes", action="store_true", help="non-interactive: accept defaults (never approves exemptions or activates policies)")
    p.add_argument("--activate", action="store_true", help="activate drafts that pass the gate (explicit; --yes alone never activates)")
    p.add_argument("--mode", choices=["log", "block"], help="default enforcement mode for all agents (default: ask; log with --yes)")
    p.add_argument("--strategy", choices=["recommended", "choose", "templates"],
                   help="policies: keep the ones agents already use + templates (recommended), choose per agent, "
                        "or templates for all (default: ask; templates with --yes)")
    p.add_argument("--restart", action="store_true", help="start the flow from step 1")
    p.add_argument("--stop-after", metavar="STEP", help=argparse.SUPPRESS)
    p.add_argument("--agent", metavar="ID", help="limit policy and mapping steps to one agent")
    _shared(p)


def _scan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print the inventory as JSON")
    p.add_argument("--check", action="store_true", help="CI mode: exit 3 on findings at --fail-on level or vocabulary drift")
    p.add_argument("--fail-on", choices=["high", "medium", "low"], default="high", help="finding level that fails --check (default: high)")
    p.add_argument("--compact", action="store_true", help="header, agent counts, coverage and finding counts only")
    p.add_argument("--no-save", action="store_true", help="do not write the report into the workspace")
    p.add_argument("--budget", type=float, default=120.0, metavar="SECONDS", help="time budget; results are marked partial when exceeded")
    p.add_argument("--yes", action="store_true", help="confirm --probe without asking")
    p.add_argument("--share", action="store_true",
                   help="also write a card of the reach map to post (SVG, and PNG when a converter is installed); "
                        "no host name, user names or paths on it")
    p.add_argument("--anonymize", action="store_true", help="with --share: agent names replaced by their kind")


def register(sub) -> None:
    s = sub.add_parser("setup", help=SETUP_HELP, description=SETUP_HELP + ".",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    _setup_args(s)
    s.set_defaults(func=_lazy("chimera_core.venom.setup.cmd_setup"))

    # cslcore venom: the discovery scan (read-only), plus the report of the latest scan
    v = sub.add_parser("venom", help="Discover the AI agents on this host and what guards them (read-only scan)",
                       description=VENOM_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    _scan_args(v)
    _shared(v)
    v.set_defaults(func=_lazy("chimera_core.venom.commands.cmd_scan"))
    vs = v.add_subparsers(dest="venom_cmd", metavar="<command>")
    rp = vs.add_parser("report", help="latest report, or one agent in detail")
    rp.add_argument("--agent", metavar="ID", help="agent id or name")
    rp.add_argument("--format", choices=["screen", "md", "json"], default="screen")
    rp.add_argument("--rescan", action="store_true", help="scan again instead of reading the latest snapshot")
    _shared(rp)
    rp.set_defaults(func=_lazy("chimera_core.venom.commands.cmd_report"))
    mv = vs.add_parser("map", help="the reach map, full screen: select, dive into an agent, 3D sphere")
    mv.add_argument("--rescan", action="store_true", help="scan again instead of reading the latest snapshot")
    mv.add_argument("--once", action="store_true", help="print one frame and exit (for scripts and CI)")
    _shared(mv)
    mv.set_defaults(func=_lazy("chimera_core.venom.commands.cmd_map_view"))

    po = sub.add_parser("policy", help="Policy workbench: list, show, new, edit, extend, fix, verify, diff, activate")
    po.add_argument("action", choices=["list", "show", "new", "edit", "extend", "fix", "verify", "diff", "activate", "bind", "unbind"])
    po.add_argument("target", nargs="?", help="policy name or path, or agent id for `new` / `extend`")
    po.add_argument("--agent", metavar="ID", action="append", help="agent(s) the action is for (bind: repeatable)")
    po.add_argument("--match", metavar="PATTERN", help="bind: agents whose key matches, e.g. 'pay-*'")
    po.add_argument("--unbound", action="store_true", help="bind: every agent that has no policy yet")
    po.add_argument("--exec-mode", choices=["allowlist", "block"], default="allowlist", help="template for EXEC tools (default: allowlist)")
    po.add_argument("--all", action="store_true", help="new: draft for every agent that needs a policy")
    po.add_argument("--yes", action="store_true", help="confirm without asking")
    _shared(po)
    po.set_defaults(func=_lazy("chimera_core.venom.policy.workbench.cmd_policy"))

    mp = sub.add_parser("map", help="Mapping assistant and mapping test (fail-open detection)")
    mp.add_argument("--agent", metavar="ID", help="agent to map")
    mp.add_argument("--policy", metavar="PATH", help="policy to map against (default: the agent's active policy)")
    mp.add_argument("--mapping", metavar="PATH[:FUNC]",
                    help="test your own mapping: a module with map_call, path.py:function (LangChain context_mapper and "
                         "plain functions are adapted), or 'openclaw' for the built-in OpenClaw map_context")
    mp.add_argument("--test", action="store_true", help="run the mapping test")
    mp.add_argument("--import-module", action="store_true",
                    help="with path.py:function, import the whole file (runs its top-level code) instead of only the function")
    mp.add_argument("--allowed-root", action="append", metavar="PATH",
                    help="bypass tests: a folder your mapping accepts as in scope (repeatable)")
    mp.add_argument("--allowed-command", action="append", metavar="CMD",
                    help="bypass tests: a command your mapping accepts as allowlisted (repeatable)")
    mp.add_argument("--allowed-destination", action="append", metavar="URL|ADDRESS",
                    help="bypass tests: a destination your mapping accepts (repeatable)")
    mp.add_argument("--classify", action="append", metavar="VAR=KIND[:PARAM]",
                    help="bypass tests for your own variable names: KIND is scope, command or destination, "
                         "PARAM the tool parameter it comes from (e.g. path_ok=scope:file_path)")
    mp.add_argument("--cases", metavar="FILE",
                    help="regression cases (JSON lines: tool, args, context, expect BLOCK or ALLOW, note)")
    mp.add_argument("--keep-cases", action="store_true",
                    help="with --cases: keep them in .csl/venom/cases/ so every later mapping test runs them")
    mp.add_argument("--yes", action="store_true", help="write the generated mapping without asking")
    _shared(mp)
    mp.set_defaults(func=_lazy("chimera_core.venom.mapping.assistant.cmd_map"))

    ex = sub.add_parser("exempt", help="Exemptions for fully trusted agents or tools: add, list, approve, remove")
    ex.add_argument("action", choices=["add", "list", "approve", "remove"])
    ex.add_argument("target", nargs="?", help="agent id for add; exemption number for approve / remove")
    ex.add_argument("--scope", choices=["agent", "tool"], default="agent")
    ex.add_argument("--tool", metavar="NAME", help="tool name (scope tool)")
    ex.add_argument("--reason", help="why this agent or tool is trusted (required)")
    ex.add_argument("--approved-by", help="who takes responsibility (required)")
    ex.add_argument("--expires", metavar="YYYY-MM-DD", help="optional expiry date")
    ex.add_argument("--propose", action="store_true", help="record as proposed; approve later")
    _shared(ex, scan=False)
    ex.set_defaults(func=_lazy("chimera_core.venom.exempt_cmd.cmd_exempt"))

    mo = sub.add_parser("mode", help="Per-agent enforcement mode (log or block) and kill switches")
    mo.add_argument("--agent", metavar="ID", help="agent key (as in `cslcore watch`); omit to list every agent")
    mo.add_argument("--all", action="store_true", help="apply the mode to every agent (sets the default, clears per-agent modes)")
    mo.add_argument("--match", metavar="PATTERN", help="apply to agents whose key matches, e.g. 'pay-*'")
    mo.add_argument("--yes", action="store_true", help="confirm --all without asking")
    mo.add_argument("mode", nargs="?", choices=["log", "block"], help="omit to show the current state")
    mo.add_argument("--disable", action="store_true", help="kill switch: block every action of the agent, in any mode")
    mo.add_argument("--enable", action="store_true", help="lift the kill switch")
    mo.add_argument("--disable-tool", metavar="TOOL", action="append", default=[], help="block one tool of the agent (repeatable)")
    mo.add_argument("--enable-tool", metavar="TOOL", action="append", default=[], help="allow a disabled tool again")
    _shared(mo, scan=False)
    mo.set_defaults(func=_lazy("chimera_core.venom.exempt_cmd.cmd_mode"))

    st = sub.add_parser("studio", help="Write, prove (Z3, TLA+) and ship CSL policies in a full-screen editor")
    st.add_argument("policy", nargs="?", help="policy file to open (default: pick one)")
    st.add_argument("--agent", metavar="ID", help="open (or start) the policy of this agent")
    st.add_argument("--new", action="store_true", help="start a new policy")
    st.add_argument("--mock", action="store_true", help="TLA+ with the Python model checker even when TLC is available")
    st.add_argument("--workspace", metavar="PATH", help="Venom workspace (default: current folder)")
    st.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)
    st.add_argument("--plan-only", action="store_true", help=argparse.SUPPRESS)
    st.set_defaults(func=_lazy("chimera_core.venom.studio.command.cmd_studio"))

    hk = sub.add_parser("hook", help="Claude Code PreToolUse hook backed by a verified policy (reads the event on stdin)")
    hk.add_argument("--agent", required=True, metavar="ID", help="agent id used in decision logs and `cslcore mode`")
    hk.add_argument("--policy", metavar="PATH", help="policy path, relative to the workspace (default: the agent's binding)")
    hk.add_argument("--mapping", metavar="PATH", help="mapping module path (default: the agent's binding)")
    hk.add_argument("--mode", choices=["log", "block"], help="override the mode set with `cslcore mode`")
    hk.add_argument("--workspace", metavar="PATH", help="Venom workspace (default: current folder)")
    hk.set_defaults(func=_lazy("chimera_core.venom.hook.cmd_hook"))

    wa = sub.add_parser("watch", help="Live management panel: decisions, log / block per agent, kill switches, exemptions, rule tuning")
    wa.add_argument("--once", action="store_true", help="render one frame and exit (for scripts and CI)")
    wa.add_argument("--refresh", type=float, default=1.0, metavar="SECONDS")
    _shared(wa, scan=False)
    wa.set_defaults(func=_lazy("chimera_core.venom.exempt_cmd.cmd_watch"))

    # help lists the 0.6 commands in the order a first install uses them
    journey = ["venom", "setup", "studio", "watch", "policy", "map", "mode", "exempt", "hook"]
    actions = getattr(sub, "_choices_actions", None)
    if isinstance(actions, list):
        rank = {name: i for i, name in enumerate(journey)}
        new = sorted((a for a in actions if a.dest in rank), key=lambda a: rank[a.dest])
        actions[:] = [a for a in actions if a.dest not in rank] + new
