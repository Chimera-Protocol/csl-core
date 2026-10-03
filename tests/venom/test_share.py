"""`cslcore venom --share`: a card to post, with nothing that identifies the machine."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from chimera_core.venom.render import share
from chimera_core.venom.render.theme import THEME

from .conftest import HOST_OPS, run_cli, scan_fixture


@pytest.fixture(scope="module")
def inv(tmp_path_factory):
    return scan_fixture(HOST_OPS, tmp_path_factory.mktemp("share")).inventory


def _text(inv, anonymize=False):
    c = Console(theme=THEME, width=share.CARD_WIDTH, file=io.StringIO(), record=True)
    c.print(share.card(inv, "0.6.3", anonymize=anonymize))
    return c.export_text()


def test_card_tells_the_story_without_identifying_the_machine(inv):
    text = _text(inv)
    assert "what can reach what on this machine" in text and "1 reach chain" in text
    assert "STRONGEST CHAIN" in text and "root on the host" in text and "pip install csl-core" in text
    assert inv.host.name not in text and "/srv" not in text and "/webhooks" not in text


def test_anonymize_hides_agent_names(inv):
    text = _text(inv, anonymize=True)
    assert "claude-code:ops" not in text and "membership-bot" not in text and "assistant 1" in text
    assert any(a.display_name == "claude-code:ops" for a in inv.agents)  # the scan itself is not changed


def test_share_flag_writes_the_card(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim", "--share"], capsys)
    assert rc == 0 and "SHARE" in out
    svgs = list((ws / ".csl" / "venom" / "share").glob("venom-*.svg"))
    assert len(svgs) == 1
    svg = svgs[0].read_text()
    assert "<svg" in svg and "/srv" not in svg
    rc, out, _ = run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim", "--share",
                          "--plan-only"], capsys)
    assert "no share card written" in out
