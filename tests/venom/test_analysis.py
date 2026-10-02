"""B8 coverage and drift, B9 risk and findings, B10 exemptions, B11 screen, B12 reports."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chimera_core.venom import model as M
from chimera_core.venom.analysis import findings as F
from chimera_core.venom.analysis.coverage import analyze, blocking_drift
from chimera_core.venom.analysis.risk import classify
from chimera_core.venom.resolve import Listener

from .conftest import HOST_EMPTY, HOST_OPS, render, run_cli, scan_fixture

GOLDEN = Path(__file__).parent / "golden"
NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def by_name(inv, name):
    return inv.agent(name)


# ---------------------------------------------------------------------------
# B8
# ---------------------------------------------------------------------------

def test_b8_enum_drift_with_suggestion(ops_inv):
    items = [d for d in ops_inv.drift if d.kind == "unknown_value" and d.value == "TRANSFER_FUNDS"]
    assert len(items) == 1 and items[0].suggestion == "transfer_funds"


def test_b8_coercion_bool_to_enum(ops_inv):
    items = [d for d in ops_inv.drift if d.kind == "coercion" and d.variable == "requires_dual_approval"]
    assert len(items) == 1 and "to_flag" in items[0].detail


def test_b8_coverage_numbers(ops_inv):
    total = sum(len(a.tools) for a in ops_inv.agents)
    c = ops_inv.coverage
    assert c.tools_total == total
    assert c.guarded + c.wired_no_rule + c.unguarded + c.exempt == total
    # hand count: wired agents are claude-code:ops (hook) and membership-bot (wrapper), no rule matches yet
    wired = sum(len(a.tools) for a in ops_inv.agents if a.guard.status != "none")
    assert c.wired_no_rule == wired and c.guarded == 0


def test_b8_check_exit_codes(tmp_path, capsys):
    rc, *_ = run_cli(["venom", "--root", str(HOST_OPS), "--check", "--no-save", "--workspace", str(tmp_path)], capsys)
    assert rc == 3
    rc, *_ = run_cli(["venom", "--root", str(HOST_EMPTY), "--check", "--no-save", "--workspace", str(tmp_path)], capsys)
    assert rc == 0


def test_b8_guarded_after_fix():
    a = M.Agent(id="code:/x", display_name="x", kind="code",
                tools=[M.Tool("transfer_funds", "decorator", [M.ToolParam("amount", "int")], risk_class="SPEND")],
                guard=M.Guard(status="wired", policy_ids=["p.csl"]))
    p = M.PolicyRef(path="/x/p.csl", status="active", domain="P", variables={"tool": '{"transfer_funds"}', "amount": "0..10"},
                    vocabulary={"tool": ["transfer_funds"]}, rules=["r"], rule_values={"r": ["tool=transfer_funds", "amount"]})
    cov, drift, _ = analyze([a], [p])
    assert cov.guarded == 1 and not blocking_drift(drift)


# ---------------------------------------------------------------------------
# B9
# ---------------------------------------------------------------------------

def test_b9_unclassified_is_sensitive():
    cls, _ = classify(M.Tool("frobnicate", "decorator"))
    assert cls == "UNCLASSIFIED" and cls in M.SENSITIVE


@pytest.mark.parametrize("name,desc,calls,expected", [
    ("run_shell", None, [], "EXEC"),
    ("helper", None, ["subprocess.run"], "EXEC"),
    ("universe_info", None, ["counts.update"], "READ"),
    ("scaffold_policy", "Generate a policy from a description.", [], "READ"),
    ("send_invoice", None, [], "SPEND"),
    ("post_to_page", None, [], "EXTERNAL"),
    ("drop_table", None, [], "DESTRUCTIVE"),
    ("x", "Deletes every record for the user", [], "DESTRUCTIVE"),
])
def test_b9_classification(name, desc, calls, expected):
    assert classify(M.Tool(name, "decorator", description=desc), calls)[0] == expected


def _agent(**kw):
    base = dict(id="code:/a", display_name="a", kind="code")
    base.update(kw)
    return M.Agent(**base)


def _inv(agents=(), policies=(), drift=(), layers=("code", "config", "triggers", "runtime", "history", "policies")):
    return M.Inventory(host=M.Host(layers_run=list(layers)), agents=list(agents), policies=list(policies), drift=list(drift))


def _ctx(**kw):
    c = {"home": "/home/op", "now": NOW, "listeners": [], "links": {}, "expired": [], "state": {}}
    c.update(kw)
    return c


EXEC_UNGUARDED = M.Tool("sh", "decorator", risk_class="EXEC", coverage="unguarded")
EXEC_GUARDED = M.Tool("sh", "decorator", risk_class="EXEC", coverage="guarded")

RULE_CASES = {
    "V01": (lambda: (_inv([_agent(access=M.Access(elevated=True))]), _ctx()), lambda: (_inv([_agent()]), _ctx())),
    "V02": (lambda: (_inv([_agent(tools=[EXEC_UNGUARDED])]), _ctx()), lambda: (_inv([_agent(tools=[EXEC_GUARDED])]), _ctx())),
    "V03": (lambda: (_inv([_agent(kind="assistant", access=M.Access(permission_mode="bypass"))]), _ctx()),
            lambda: (_inv([_agent(kind="assistant", access=M.Access(permission_mode="default"))]), _ctx())),
    "V04": (lambda: (_inv([_agent(triggers=[M.Trigger("messaging", "/sms")], tools=[M.Tool("post", "d", risk_class="EXTERNAL", coverage="unguarded")])]), _ctx()),
            lambda: (_inv([_agent(triggers=[M.Trigger("time", "daily")], tools=[M.Tool("post", "d", risk_class="EXTERNAL", coverage="unguarded")])]), _ctx())),
    "V05": (lambda: (_inv([_agent(triggers=[M.Trigger("time", "daily")], tools=[M.Tool("w", "d", risk_class="WRITE", coverage="unguarded")])]), _ctx()),
            lambda: (_inv([_agent(triggers=[M.Trigger("time", "daily")], tools=[M.Tool("r", "d", risk_class="READ", coverage="unguarded")])]), _ctx())),
    "V06": (lambda: (_inv([_agent(access=M.Access(credentials=[M.Credential("AWS_SECRET_ACCESS_KEY", "/.env", "cloud_root")]))]), _ctx()),
            lambda: (_inv([_agent(access=M.Access(credentials=[M.Credential("OPENAI_API_KEY", "/.env", "llm_key")]))]), _ctx())),
    "V07": (lambda: (_inv([_agent(access=M.Access(fs_roots=["/home/op"]))]), _ctx()),
            lambda: (_inv([_agent(access=M.Access(fs_roots=["/home/op/project"]))]), _ctx())),
    "V08": (lambda: (_inv(), _ctx(listeners=[Listener(1, "mcp-server-x", ["0.0.0.0:80"], True)])),
            lambda: (_inv(), _ctx(listeners=[Listener(1, "mcp-server-x", ["127.0.0.1:80"], True)]))),
    "V09": (lambda: (_inv([_agent(tools=[M.Tool("pay", "d", risk_class="SPEND", coverage="wired_no_rule")])]), _ctx()),
            lambda: (_inv([_agent(tools=[M.Tool("pay", "d", risk_class="SPEND", coverage="guarded")])]), _ctx())),
    "V10": (lambda: (_inv(drift=[M.DriftItem("unknown_value", "P", "tool", "X", "x")]), _ctx()),
            lambda: (_inv(drift=[M.DriftItem("unsupplied_variable", "P", "user_role")]), _ctx())),
    "V11": (lambda: (_inv(), _ctx(state={"mapping_tests": {"a": {"fail_open": 2}}})),
            lambda: (_inv(), _ctx(state={"mapping_tests": {"a": {"fail_open": 0}}}))),
    "V12": (lambda: (_inv(policies=[M.PolicyRef("/w/policies/p.csl", "active", domain="P")]), _ctx()),
            lambda: (_inv(policies=[M.PolicyRef("/w/policies/p.csl", "active", domain="P")]), _ctx(links={"x": ["/w/policies/p.csl"]}))),
    "V13": (lambda: (_inv(), _ctx(expired=[M.Exemption("a", reason="r", approved_by="me", expires="2026-01-01", status="approved")])),
            lambda: (_inv(), _ctx())),
    "V14": (lambda: (_inv([_agent(kind="unmanaged")]), _ctx()), lambda: (_inv([_agent(kind="code")]), _ctx())),
    "V15": (lambda: (_inv([_agent(state="scheduled", triggers=[M.Trigger("time", "daily")], runs=M.RunStats(0, "7d", "2026-08-01"))]), _ctx()),
            lambda: (_inv([_agent(state="scheduled", triggers=[M.Trigger("time", "daily")], runs=M.RunStats(1, "7d", "2026-09-30"))]), _ctx())),
    "V16": (lambda: (_inv(), _ctx(state={"modes": {"a": {"mode": "log", "since": "2026-09-01T00:00:00+00:00"}}})),
            lambda: (_inv(), _ctx(state={"modes": {"a": {"mode": "log", "since": "2026-09-28T00:00:00+00:00"}}}))),
}


@pytest.mark.parametrize("rule", sorted(F.RULES))
def test_b9_rule_positive_and_negative(rule):
    pos, neg = RULE_CASES[rule]
    inv, ctx = pos()
    found, _ = F.evaluate(inv, ctx)
    assert any(f.id == rule for f in found), f"{rule} did not fire"
    inv, ctx = neg()
    found, _ = F.evaluate(inv, ctx)
    assert not any(f.id == rule for f in found), f"{rule} fired on the negative case"


def test_b9_skipped_layer_reported(tmp_path):
    from chimera_core.venom.render.report import to_markdown

    inv = _inv([_agent(access=M.Access(elevated=True))], layers=("code", "config", "policies"))
    found, not_eval = F.evaluate(inv, _ctx())
    assert "V01" in not_eval and not any(f.id == "V01" for f in found)
    inv.rules_not_evaluated = not_eval
    assert "Rules not evaluated" in to_markdown(inv)


# ---------------------------------------------------------------------------
# B10
# ---------------------------------------------------------------------------

def test_b10_add_requires_reason_and_approver(tmp_path, capsys):
    rc, out, _ = run_cli(["exempt", "add", "code:/srv/ingest-worker", "--workspace", str(tmp_path)], capsys)
    assert rc == 2 and "--reason" in out
    rc, out, _ = run_cli(["exempt", "add", "code:/srv/ingest-worker", "--reason", "x", "--workspace", str(tmp_path)], capsys)
    assert rc == 2 and "--approved-by" in out


def test_b10_approved_exemption_moves_findings(tmp_path, capsys):
    run_cli(["exempt", "add", "code:/srv/ingest-worker", "--reason", "Isolated batch host", "--approved-by", "aytug",
             "--workspace", str(tmp_path)], capsys)
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    w = by_name(inv, "ingest-worker")
    assert w.exempt is not None
    assert not [f for f in inv.findings if f.agent_id == w.id]
    assert [f for f in inv.exempted if f.agent_id == w.id]
    assert inv.coverage.exempt == len(w.tools)
    from chimera_core.venom.render.report import to_markdown
    md = to_markdown(inv)
    assert "Exempted by operator" in md and "Isolated batch host" in md


def test_b10_expired_exemption_raises_v13(tmp_path, capsys):
    run_cli(["exempt", "add", "code:/srv/ingest-worker", "--reason", "old", "--approved-by", "aytug",
             "--expires", "2026-01-01", "--workspace", str(tmp_path)], capsys)
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    assert by_name(inv, "ingest-worker").exempt is None
    assert any(f.id == "V13" for f in inv.findings)


def test_b10_proposed_has_no_effect_until_approved(tmp_path, capsys):
    run_cli(["exempt", "add", "code:/srv/ingest-worker", "--reason", "trusted", "--propose", "--workspace", str(tmp_path)], capsys)
    assert by_name(scan_fixture(HOST_OPS, tmp_path).inventory, "ingest-worker").exempt is None
    rc, out, _ = run_cli(["exempt", "approve", "1", "--workspace", str(tmp_path)], capsys)
    assert rc == 2  # approval needs an approver
    rc, out, _ = run_cli(["exempt", "approve", "1", "--approved-by", "aytug", "--workspace", str(tmp_path)], capsys)
    assert rc == 0
    assert by_name(scan_fixture(HOST_OPS, tmp_path).inventory, "ingest-worker").exempt is not None


def test_b10_yaml_round_trip():
    from chimera_core.venom import exemptions as ex
    items = [M.Exemption("assistant:claude-code:/srv/ops", "agent", None, 'Operator\'s "own" session', "aytug", "2026-12-31", "approved"),
             M.Exemption("*", "tool", "read_file", "Read-only", "aytug", None, "approved")]
    assert ex.parse(ex.dump(items)) == items


# ---------------------------------------------------------------------------
# B11 screen
# ---------------------------------------------------------------------------

def _screen(inv, width):
    from chimera_core.venom.render.screen import scan_screen
    inv.host.duration_ms = 1234
    return render(scan_screen(inv, "0.6.0", width, report_hint=".csl/venom/reports/latest.md"), width=width)


@pytest.mark.parametrize("width", [100, 80])
def test_b11_golden_snapshots(ops_inv, width):
    text = _screen(ops_inv, width)
    path = GOLDEN / f"scan_ops_{width}.txt"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("width", [52, 64, 80])
def test_b11_fits_narrow_terminals(ops_inv, width):
    text = _screen(ops_inv, width)
    for line in text.splitlines():
        assert len(line.rstrip()) <= width, line
    assert "ingest-worker" in text and "COVERAGE" in text


def test_b11_no_color_no_escapes(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    rc, out, _ = run_cli(["venom", "--root", str(HOST_OPS), "--no-save", "--workspace", str(tmp_path)], capsys)
    assert "\x1b[" not in out


def test_b11_color_has_escapes(ops_inv):
    from chimera_core.venom.render.screen import scan_screen
    assert "\x1b[" in render(scan_screen(ops_inv, "0.6.0", 100), color=True)


def test_b11_at_most_40_lines(ops_inv):
    assert len(_screen(ops_inv, 100).splitlines()) <= 40


def test_b11_compact(ops_inv):
    from chimera_core.venom.render.screen import scan_screen
    text = render(scan_screen(ops_inv, "0.6.0", 100, compact=True))
    assert "AGENTS" in text and "COVERAGE" in text and "Agent " not in text and len(text.splitlines()) <= 10


def test_b11_agent_detail(ops_inv):
    from chimera_core.venom.render.screen import agent_detail
    w = by_name(ops_inv, "ingest-worker")
    text = render(agent_detail(ops_inv, w, "0.6.0", 100))
    assert "run_command" in text and "EXEC" in text and "OPENAI_API_KEY" in text and "root" in text
    assert re.search(r"7 in 7d\s+[·▁▂▃▄▅▆▇█]{7}", text)


# ---------------------------------------------------------------------------
# B12 reports
# ---------------------------------------------------------------------------

def test_b12_json_validates(ops_inv):
    jsonschema = pytest.importorskip("jsonschema")
    from chimera_core.venom.render.report import to_json
    schema = json.loads((Path(__file__).resolve().parents[2] / "chimera_core/venom/schema/report.v1.json").read_text())
    jsonschema.validate(to_json(ops_inv), schema)


def test_b12_markdown_complete(ops_inv):
    from chimera_core.venom.render.report import to_markdown
    md = to_markdown(ops_inv)
    for a in ops_inv.agents:
        assert a.display_name in md
    for f in ops_inv.findings:
        assert f.summary.replace("|", "\\|") in md
    for d in ops_inv.drift:
        assert d.variable in md
    assert "## Exempted by operator" in md


def test_b12_latest_equals_newest(tmp_path, capsys):
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(tmp_path)], capsys)
    reports = tmp_path / ".csl/venom/reports"
    newest_json = sorted(reports.glob("report-*.json"))[-1]
    newest_md = sorted(reports.glob("report-*.md"))[-1]
    assert (reports / "latest.json").read_text() == newest_json.read_text()
    assert (reports / "latest.md").read_text() == newest_md.read_text()


def test_b12_redaction_of_inline_secrets():
    from chimera_core.venom import redact
    s = redact.text("OPENAI_API_KEY=sk-abc123456789012345 python a.py --token hunter2 postgres://u:pw@db/x Bearer abc.def")
    assert "sk-abc" not in s and "hunter2" not in s and ":pw@" not in s and "abc.def" not in s
