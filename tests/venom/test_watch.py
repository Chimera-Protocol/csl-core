"""B19 watch dashboard."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chimera_core.venom import watch as W
from chimera_core.venom.workspace import Workspace

from .conftest import render, run_cli

DECISIONS = Path(__file__).parent / "fixtures" / "decisions"
GOLDEN = Path(__file__).parent / "golden"
NOW = datetime(2026, 10, 1, 14, 28, tzinfo=timezone.utc)


def _ws(tmp_path) -> Workspace:
    ws = Workspace(tmp_path)
    shutil.copytree(DECISIONS, ws.decisions, dirs_exist_ok=True)
    return ws


def _model(ws):
    m = W.WatchModel()
    W.Tail(ws).poll(m)
    return m


def _frame(ws, width):
    m = _model(ws)
    return render(W.render(m, None, {}, {"ingest-worker": "log"}, NOW, NOW - timedelta(hours=2, minutes=14), width, 30), width=width, height=30)


@pytest.mark.parametrize("width", [120, 80])
def test_b19_golden_layout(tmp_path, width):
    text = _frame(_ws(tmp_path), width)
    path = GOLDEN / f"watch_{width}.txt"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")
    assert all(len(l) <= width for l in text.splitlines())
    # the live stream is newest first across all agents
    assert "14:27:53" in text


def test_b19_tuning_ranking(tmp_path):
    import json
    from collections import Counter
    expected = Counter()
    for f in DECISIONS.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            for r in json.loads(line)["rules"]:
                expected[r] += 1
    ranked = [(rule, rs.total) for rule, rs in W.ranked_rules(_model(_ws(tmp_path)))]
    assert ranked == sorted(expected.items(), key=lambda kv: (-kv[1], kv[0]))
    rs = dict(W.ranked_rules(_model(_ws(tmp_path))))
    assert rs["no_live_service_writes"].agents == {"ingest-worker", "publisher"}


def test_b19_empty_state(tmp_path, capsys):
    rc, out, _ = run_cli(["watch", "--once", "--workspace", str(tmp_path), "--no-color"], capsys)
    assert rc == 0 and "no decisions yet" in out and "cslcore setup" in out


def test_b19_incremental_tail_and_bounded_memory(tmp_path):
    ws = Workspace(tmp_path)
    m = W.WatchModel()
    tail = W.Tail(ws)
    log = ws.decision_log("load")
    log.parent.mkdir(parents=True)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    # 10 minutes at 100 decisions per second, appended in batches like a live agent
    for second in range(600):
        batch = "".join(
            f'{{"ts": "{(start + timedelta(seconds=second, milliseconds=i * 10)).isoformat()}", "agent": "load", "tool": "t", '
            f'"decision": "{"WOULD_BLOCK" if i % 10 == 0 else "ALLOW"}", "rules": {["r1"] if i % 10 == 0 else []}}}\n'.replace("'", '"')
            for i in range(100))
        with open(log, "a") as f:
            f.write(batch)
        if second % 60 == 0:
            tail.poll(m)
    tail.poll(m)
    assert m.total == 60_000
    assert len(m.stream) == W.STREAM_SIZE
    assert m.rules["r1"].total == 6_000
    m.per_minute(start + timedelta(minutes=10))
    assert len(m.minute) <= 100_000


def test_b19_partial_line_is_not_lost(tmp_path):
    ws = Workspace(tmp_path)
    log = ws.decision_log("a")
    log.parent.mkdir(parents=True)
    log.write_text('{"ts": "2026-10-01T00:00:00+00:00", "agent": "a", "decision": "ALLOW"}\n{"ts": "2026-10-01T00:00:01+00:00", "ag')
    m = W.WatchModel()
    tail = W.Tail(ws)
    tail.poll(m)
    assert m.total == 1
    with open(log, "a") as f:
        f.write('ent": "a", "decision": "BLOCK", "rules": ["x"]}\n')
    tail.poll(m)
    assert m.total == 2 and m.agents["a"].block == 1
