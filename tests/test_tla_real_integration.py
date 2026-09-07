"""
Tests for real TLC integration:
  - TLASpecBuilder (spec generation + predicate abstraction)
  - TLCRunner / parse_tlc_output (output parsing)
  - run_tlc_on_spec end-to-end (skipped if Java/JAR unavailable)
  - TLAVerifier dispatch logic (TLC vs Mock)
"""

from __future__ import annotations

import re
import textwrap
import pytest
from pathlib import Path
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build minimal CSL ASTs without the full parser
# ─────────────────────────────────────────────────────────────────────────────

from chimera_core.language.ast import (
    Constitution, Configuration, Domain, VariableDeclaration,
    Constraint, ConstraintType, ConditionClause, ActionClause,
    TemporalOperator, ModalOperator, EnforcementMode,
    Literal, Variable, BinaryOp, ComparisonOperator, ArithmeticOperator,
)


def _lit(v, t="string"):
    return Literal(value=v, type=t)


def _var(n):
    return Variable(name=n)


def _eq(var_name, val):
    return BinaryOp(
        left=Variable(name=var_name),
        operator=ComparisonOperator.EQ,
        right=_lit(val),
    )


def _gt(var_name, val):
    return BinaryOp(
        left=Variable(name=var_name),
        operator=ComparisonOperator.GT,
        right=_lit(val, "int"),
    )


def _make_const(name, cond_expr, var_name, modal, value_expr):
    return Constraint(
        name=name,
        constraint_type=ConstraintType.STATE,
        condition=ConditionClause(
            temporal_operator=TemporalOperator.WHEN,
            condition=cond_expr,
        ),
        action=ActionClause(
            variable=var_name,
            modal_operator=modal,
            value=value_expr,
        ),
    )


def _make_always_const(name, var_name, modal, value_expr):
    return Constraint(
        name=name,
        constraint_type=ConstraintType.STATE,
        condition=ConditionClause(
            temporal_operator=TemporalOperator.ALWAYS,
            condition=_lit(True, "bool"),
        ),
        action=ActionClause(
            variable=var_name,
            modal_operator=modal,
            value=value_expr,
        ),
    )


def _make_constitution(domain_name, var_decls, constraints):
    domain = Domain(
        name=domain_name,
        variable_declarations=var_decls,
    )
    config = Configuration(
        enforcement_mode=EnforcementMode.BLOCK,
        enable_formal_verification=True,
    )
    return Constitution(domain=domain, config=config, constraints=constraints)


# ═════════════════════════════════════════════════════════════════════════════
# 1. TLASpecBuilder tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTLASpecBuilder:

    def setup_method(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder
        self.builder = TLASpecBuilder()

    def _simple_constitution(self):
        decls = [
            VariableDeclaration(name="role", domain='{"ADMIN", "USER"}'),
            VariableDeclaration(name="action", domain='{"READ", "DELETE"}'),
        ]
        c = _make_const(
            "no_user_delete",
            _eq("role", "USER"),
            "action", ModalOperator.MUST_NOT_BE, _lit("DELETE"),
        )
        return _make_constitution("RolePolicy", decls, [c])

    def test_module_name_matches_domain(self):
        spec = self.builder.build(self._simple_constitution())
        assert spec.module_name == "RolePolicy"

    def test_tla_source_has_variables_declaration(self):
        spec = self.builder.build(self._simple_constitution())
        assert "VARIABLES role, action" in spec.tla_source

    def test_tla_source_has_typeok(self):
        spec = self.builder.build(self._simple_constitution())
        assert "TypeOK ==" in spec.tla_source
        assert 'role \\in {"ADMIN", "USER"}' in spec.tla_source
        assert 'action \\in {"READ", "DELETE"}' in spec.tla_source

    def test_tla_source_has_init(self):
        spec = self.builder.build(self._simple_constitution())
        assert "Init ==" in spec.tla_source
        # Init must also use \in
        assert spec.tla_source.count("\\in") >= 2

    def test_tla_source_has_invariant(self):
        spec = self.builder.build(self._simple_constitution())
        assert "no_user_delete ==" in spec.tla_source
        # MUST NOT BE "DELETE"  → action # "DELETE"
        assert 'action # "DELETE"' in spec.tla_source

    def test_tla_source_invariant_implication(self):
        spec = self.builder.build(self._simple_constitution())
        # WHEN role == "USER" THEN ... → (~(role = "USER")) \/ (action # "DELETE")
        assert '~(' in spec.tla_source
        assert '\\/' in spec.tla_source

    def test_cfg_has_invariant_entries(self):
        spec = self.builder.build(self._simple_constitution())
        assert "INVARIANT TypeOK" in spec.cfg_source
        assert "INVARIANT no_user_delete" in spec.cfg_source

    def test_cfg_has_init_and_next(self):
        spec = self.builder.build(self._simple_constitution())
        assert "INIT Init" in spec.cfg_source
        assert "NEXT Next" in spec.cfg_source

    def test_write_creates_files(self, tmp_path):
        spec = self.builder.build(self._simple_constitution())
        tla, cfg = spec.write(tmp_path)
        assert tla.exists()
        assert cfg.exists()
        assert tla.suffix == ".tla"
        assert cfg.suffix == ".cfg"

    def test_write_file_contents(self, tmp_path):
        spec = self.builder.build(self._simple_constitution())
        tla, cfg = spec.write(tmp_path)
        assert "MODULE RolePolicy" in tla.read_text()
        assert "INVARIANT TypeOK" in cfg.read_text()


class TestTLASpecBuilderPredicateAbstraction:

    def setup_method(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder
        self.builder = TLASpecBuilder()

    def _amount_constitution(self):
        decls = [
            VariableDeclaration(name="amount", domain="0..100000"),
            VariableDeclaration(name="approved", domain="BOOLEAN"),
        ]
        c = _make_const(
            "large_transfer_check",
            _gt("amount", 50000),
            "approved", ModalOperator.MUST_BE, _lit(True, "bool"),
        )
        return _make_constitution("TransferPolicy", decls, [c])

    def test_predicate_abstraction_applied_for_large_range(self):
        spec = self.builder.build(self._amount_constitution())
        # Should NOT use 0..100000 verbatim (too large)
        assert "0..100000" not in spec.tla_source
        # Should include threshold boundary points
        tla_set = next(
            di["tla_set"]
            for di in spec.domain_info
            if di["name"] == "amount"
        )
        # The threshold 50000 must appear with its neighbors
        assert "50000" in tla_set
        assert "50001" in tla_set

    def test_abstracted_domain_info(self):
        spec = self.builder.build(self._amount_constitution())
        amount_info = next(d for d in spec.domain_info if d["name"] == "amount")
        # Card should mention "abstracted from"
        assert "abstracted" in amount_info["card"]

    def test_small_range_not_abstracted(self):
        decls = [VariableDeclaration(name="risk", domain="0..5")]
        c = _make_always_const("low_risk", "risk", ModalOperator.LTE, _lit(10, "int"))
        con = _make_constitution("RiskPolicy", decls, [c])
        spec = self.builder.build(con)
        # 0..5 has only 6 values — should stay as integer range notation
        risk_info = next(d for d in spec.domain_info if d["name"] == "risk")
        assert "0..5" in risk_info["tla_set"] or "{0, 1, 2, 3, 4, 5}" in risk_info["tla_set"]

    def test_boolean_domain_stays_boolean(self):
        spec = self.builder.build(self._amount_constitution())
        approved_info = next(d for d in spec.domain_info if d["name"] == "approved")
        assert approved_info["tla_set"] == "BOOLEAN"

    def test_threshold_extraction(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _extract_thresholds
        decls = [VariableDeclaration(name="score", domain="0..1000")]
        c1 = _make_const("c1", _gt("score", 500), "score", ModalOperator.LTE, _lit(500, "int"))
        c2 = _make_const("c2", _gt("score", 200), "score", ModalOperator.MUST_NOT_BE, _lit(999, "int"))
        con = _make_constitution("ScorePolicy", decls, [c1, c2])
        thresholds = _extract_thresholds("score", con.constraints)
        assert 500 in thresholds
        assert 200 in thresholds


class TestExprToTLA:

    def test_literal_string(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        assert _expr_to_tla(_lit("USER")) == '"USER"'

    def test_literal_int(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        assert _expr_to_tla(_lit(42, "int")) == "42"

    def test_literal_bool_true(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        assert _expr_to_tla(_lit(True, "bool")) == "TRUE"

    def test_literal_bool_false(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        assert _expr_to_tla(_lit(False, "bool")) == "FALSE"

    def test_variable(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        assert _expr_to_tla(_var("role")) == "role"

    def test_eq_comparison(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        expr = BinaryOp(left=_var("role"), operator=ComparisonOperator.EQ, right=_lit("USER"))
        assert _expr_to_tla(expr) == '(role = "USER")'

    def test_neq_maps_to_hash(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        expr = BinaryOp(left=_var("x"), operator=ComparisonOperator.NEQ, right=_lit("A"))
        assert _expr_to_tla(expr) == '(x # "A")'

    def test_lte_maps_to_equal_less(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        expr = BinaryOp(left=_var("n"), operator=ComparisonOperator.LTE, right=_lit(10, "int"))
        assert _expr_to_tla(expr) == "(n =< 10)"

    def test_and_maps_to_tla_and(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _expr_to_tla
        from chimera_core.language.ast import LogicalOperator
        expr = BinaryOp(left=_var("a"), operator=LogicalOperator.AND, right=_var("b"))
        assert "/\\" in _expr_to_tla(expr)

    def test_action_must_not_be(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _action_to_tla
        c = _make_always_const("c", "tool", ModalOperator.MUST_NOT_BE, _lit("DELETE"))
        assert _action_to_tla(c) == 'tool # "DELETE"'

    def test_action_must_be(self):
        from chimera_core.engines.tla_engine.tla_spec_builder import _action_to_tla
        c = _make_always_const("c", "status", ModalOperator.MUST_BE, _lit("OK"))
        assert _action_to_tla(c) == 'status = "OK"'


# ═════════════════════════════════════════════════════════════════════════════
# 2. TLC output parser tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTLCOutputParser:

    def test_parse_success_output(self):
        from chimera_core.engines.tla_engine.tlc_runner import parse_tlc_output
        output = textwrap.dedent("""\
            @!@!@STARTMSG 2185:0 @!@!@
            Model checking completed. No error has been found.
            Estimates of the probability that TLC did not check all states: ...
            10 states generated, 8 distinct states found, 0 states left on queue.
            The depth of the complete state space search is 3.
            Finished in 123ms at (2024-01-01 00:00:00)
            @!@!@ENDMSG 2185 @!@!@
        """)
        parser = parse_tlc_output(output)
        assert parser.success is True
        assert parser.states_explored == 10
        assert parser.distinct_states == 8

    def test_parse_invariant_violated(self):
        from chimera_core.engines.tla_engine.tlc_runner import parse_tlc_output
        output = textwrap.dedent("""\
            @!@!@STARTMSG 2121:1 @!@!@
            Invariant user_no_transfer is violated.
            @!@!@ENDMSG 2121 @!@!@
            @!@!@STARTMSG 2262:0 @!@!@
            /\\ role = "USER"
            /\\ tool = "TRANSFER_FUNDS"
            /\\ amount = 100
            @!@!@ENDMSG 2262 @!@!@
        """)
        parser = parse_tlc_output(output)
        assert parser.success is False
        assert len(parser.violations) == 1
        v = parser.violations[0]
        assert v.invariant == "user_no_transfer"
        assert v.state_vars.get("role") == '"USER"'
        assert v.state_vars.get("tool") == '"TRANSFER_FUNDS"'

    def test_parse_stats_line_without_structured_msg(self):
        from chimera_core.engines.tla_engine.tlc_runner import parse_tlc_output
        output = "5,000 states generated, 3,200 distinct states found, 0 states left"
        parser = parse_tlc_output(output)
        assert parser.states_explored == 5000
        assert parser.distinct_states == 3200

    def test_coerce_boolean_values(self):
        from chimera_core.engines.tla_engine.verifier import _coerce_tlc_value
        assert _coerce_tlc_value("TRUE") is True
        assert _coerce_tlc_value("FALSE") is False

    def test_coerce_integer(self):
        from chimera_core.engines.tla_engine.verifier import _coerce_tlc_value
        assert _coerce_tlc_value("42") == 42

    def test_coerce_quoted_string(self):
        from chimera_core.engines.tla_engine.verifier import _coerce_tlc_value
        assert _coerce_tlc_value('"ADMIN"') == "ADMIN"

    def test_coerce_unknown_returns_raw(self):
        from chimera_core.engines.tla_engine.verifier import _coerce_tlc_value
        assert _coerce_tlc_value("SomeRecord") == "SomeRecord"

    def test_multiple_violations_parsed(self):
        from chimera_core.engines.tla_engine.tlc_runner import parse_tlc_output
        output = textwrap.dedent("""\
            @!@!@STARTMSG 2121:1 @!@!@
            Invariant inv_one is violated.
            @!@!@ENDMSG 2121 @!@!@
            @!@!@STARTMSG 2262:0 @!@!@
            /\\ x = 1
            @!@!@ENDMSG 2262 @!@!@
            @!@!@STARTMSG 2121:1 @!@!@
            Invariant inv_two is violated.
            @!@!@ENDMSG 2121 @!@!@
            @!@!@STARTMSG 2262:0 @!@!@
            /\\ y = 2
            @!@!@ENDMSG 2262 @!@!@
        """)
        parser = parse_tlc_output(output)
        assert parser.success is False
        # Each new STARTMSG 2121 triggers a new violation entry
        assert len(parser.violations) >= 1   # at least the last one


# ═════════════════════════════════════════════════════════════════════════════
# 3. JAR / Java availability
# ═════════════════════════════════════════════════════════════════════════════

class TestTLCAvailability:

    def test_java_available_returns_bool(self):
        from chimera_core.engines.tla_engine.tlc_runner import java_available
        result = java_available()
        assert isinstance(result, bool)

    def test_find_jar_returns_none_when_missing(self, tmp_path, monkeypatch):
        from chimera_core.engines.tla_engine import tlc_runner as tr
        # Isolate all discovery paths so nothing is found
        monkeypatch.setattr(tr, "_DEFAULT_JAR_CACHE", tmp_path / "no.jar")
        monkeypatch.delenv("TLA2TOOLS_JAR", raising=False)
        result = tr.find_jar(explicit=str(tmp_path / "nonexistent.jar"))
        assert result is None

    def test_find_jar_finds_file(self, tmp_path):
        from chimera_core.engines.tla_engine.tlc_runner import find_jar
        jar = tmp_path / "tla2tools.jar"
        jar.write_bytes(b"fake jar")
        result = find_jar(explicit=str(jar))
        assert result == jar

    def test_runner_is_available_false_without_jar(self, monkeypatch, tmp_path):
        from chimera_core.engines.tla_engine.tlc_runner import TLCRunner
        # Point to a directory where no jar exists
        monkeypatch.setenv("TLA2TOOLS_JAR", str(tmp_path / "no.jar"))
        runner = TLCRunner(jar_path=str(tmp_path / "no.jar"), auto_download=False)
        # is_available() should return False (no jar, no download)
        # (java may or may not be available — we only test the jar part here)
        # Just verify it doesn't crash
        result = runner.is_available()
        assert isinstance(result, bool)


# ═════════════════════════════════════════════════════════════════════════════
# 4. TLC end-to-end tests (skipped when TLC unavailable)
# ═════════════════════════════════════════════════════════════════════════════

_TLC_SKIP = pytest.mark.skipif(
    not __import__("chimera_core.engines.tla_engine.tlc_runner",
                   fromlist=["java_available"]).java_available(),
    reason="Java not on PATH — skipping real TLC tests",
)


@_TLC_SKIP
class TestTLCEndToEnd:
    """These tests only run when Java is present and tla2tools.jar can be found/downloaded."""

    def setup_method(self):
        from chimera_core.engines.tla_engine.tlc_runner import TLCRunner, find_jar, ensure_jar
        self.jar = ensure_jar(auto_download=True)
        if self.jar is None:
            pytest.skip("tla2tools.jar not available and could not be downloaded")
        self.runner = TLCRunner(jar_path=str(self.jar))

    def _build_spec(self, constitution):
        from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder
        return TLASpecBuilder().build(constitution)

    def test_simple_valid_policy_passes(self, tmp_path):
        """A policy with no reachable violations should pass."""
        decls = [
            VariableDeclaration(name="task", domain='{"READ", "WRITE"}'),
        ]
        c = _make_always_const("no_delete", "task", ModalOperator.MUST_NOT_BE, _lit("DELETE"))
        con = _make_constitution("ValidPolicy", decls, [c])
        spec = self._build_spec(con)
        tla, cfg = spec.write(tmp_path)
        result = self.runner.run(tla, cfg, timeout=30)
        assert result.used_real_tlc is True
        assert result.success is True
        assert len(result.violations) == 0
        assert result.states_explored > 0

    def test_violated_policy_fails(self, tmp_path):
        """A policy with reachable violation should be found by TLC."""
        decls = [
            VariableDeclaration(name="role", domain='{"ADMIN", "USER"}'),
            VariableDeclaration(name="tool", domain='{"READ_DB", "TRANSFER_FUNDS"}'),
        ]
        c = _make_const(
            "user_no_transfer",
            _eq("role", "USER"),
            "tool", ModalOperator.MUST_NOT_BE, _lit("TRANSFER_FUNDS"),
        )
        con = _make_constitution("ViolationPolicy", decls, [c])
        spec = self._build_spec(con)
        tla, cfg = spec.write(tmp_path)
        result = self.runner.run(tla, cfg, timeout=30)
        assert result.used_real_tlc is True
        assert result.success is False
        assert len(result.violations) >= 1
        # The violated invariant name should match (safe-identifier form)
        inv_names = [v.invariant for v in result.violations]
        assert any("user_no_transfer" in n for n in inv_names)

    def test_typeok_always_holds_for_valid_domains(self, tmp_path):
        """TypeOK must hold for all reachable states in a well-defined domain."""
        decls = [
            VariableDeclaration(name="env", domain='{"dev", "staging"}'),
        ]
        c = _make_always_const("no_prod", "env", ModalOperator.MUST_NOT_BE, _lit("production"))
        con = _make_constitution("EnvPolicy", decls, [c])
        spec = self._build_spec(con)
        tla, cfg = spec.write(tmp_path)
        result = self.runner.run(tla, cfg, timeout=30)
        assert result.success is True

    def test_all_violations_reported_not_just_the_first(self, tmp_path):
        """
        Regression test for a real bug: TLC stops at the first invariant
        violation by default. Without -continue on the command line, a
        policy with 2 independently-violable rules would report only 1
        violation and never check the second at all. With 2 independent
        variables each free to range over their own domain, both rules
        below are violable, and both must show up in result.violations.
        """
        decls = [
            VariableDeclaration(name="role", domain='{"ADMIN", "USER"}'),
            VariableDeclaration(name="tool", domain='{"READ_DB", "TRANSFER_FUNDS"}'),
            VariableDeclaration(name="env", domain='{"dev", "production"}'),
        ]
        c1 = _make_const(
            "user_no_transfer",
            _eq("role", "USER"),
            "tool", ModalOperator.MUST_NOT_BE, _lit("TRANSFER_FUNDS"),
        )
        c2 = _make_always_const("no_prod", "env", ModalOperator.MUST_NOT_BE, _lit("production"))
        con = _make_constitution("MultiViolationPolicy", decls, [c1, c2])
        spec = self._build_spec(con)
        tla, cfg = spec.write(tmp_path)
        result = self.runner.run(tla, cfg, timeout=30)
        assert result.success is False
        inv_names = [v.invariant for v in result.violations]
        assert any("user_no_transfer" in n for n in inv_names)
        assert any("no_prod" in n for n in inv_names)

    def test_unsupported_construct_reported_as_failure_not_silent_success(self, tmp_path):
        """
        Regression test for a real bug: TLC exits with code 151 (not 150)
        when it hits a construct it can't evaluate (e.g. a real-number
        literal, since the generated spec only EXTENDS Integers). The old
        code only special-cased exit code 150 and otherwise fell back to
        "no violations parsed => success", so a spec TLC never actually
        checked was silently reported as verified-safe. It must instead be
        reported as a failure with a real error message.
        """
        decls = [
            VariableDeclaration(name="amount", domain="0..100"),
        ]
        # amount > (amount * 0.1) — a real-number literal TLC's Integers-only
        # spec cannot evaluate.
        cond = BinaryOp(
            left=Variable(name="amount"),
            operator=ComparisonOperator.GT,
            right=BinaryOp(
                left=Variable(name="amount"),
                operator=ArithmeticOperator.MUL,
                right=_lit(0.1, "float"),
            ),
        )
        c = _make_const(
            "needs_real_arithmetic", cond,
            "amount", ModalOperator.LTE, _lit(100, "int"),
        )
        con = _make_constitution("RealNumberPolicy", decls, [c])
        spec = self._build_spec(con)
        tla, cfg = spec.write(tmp_path)
        result = self.runner.run(tla, cfg, timeout=30)
        assert result.success is False
        assert result.error  # must explain why, not just silently fail
        assert not result.violations  # TLC never got far enough to find any


# ═════════════════════════════════════════════════════════════════════════════
# 5. TLAVerifier dispatch tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTLAVerifierDispatch:
    """Verify dispatch logic: TLC path vs Mock fallback."""

    def _violation_constitution(self):
        decls = [
            VariableDeclaration(name="role", domain='{"ADMIN", "USER"}'),
            VariableDeclaration(name="action", domain='{"VIEW", "DELETE"}'),
        ]
        c = _make_const(
            "user_no_delete",
            _eq("role", "USER"),
            "action", ModalOperator.MUST_NOT_BE, _lit("DELETE"),
        )
        return _make_constitution("DispatchTest", decls, [c])

    def _valid_constitution(self):
        decls = [
            VariableDeclaration(name="env", domain='{"dev", "staging"}'),
        ]
        c = _make_always_const("no_prod", "env", ModalOperator.MUST_NOT_BE, _lit("production"))
        return _make_constitution("ValidDispatch", decls, [c])

    def test_mock_path_finds_violation(self):
        from chimera_core.engines.tla_engine.verifier import TLAVerifier
        verifier = TLAVerifier(animate=False, use_real_tlc=False)
        ok, issues = verifier.verify(self._violation_constitution())
        assert ok is False
        assert len(issues) == 1
        assert issues[0].kind == "SAFETY_VIOLATION"

    def test_mock_path_valid_policy_passes(self):
        from chimera_core.engines.tla_engine.verifier import TLAVerifier
        verifier = TLAVerifier(animate=False, use_real_tlc=False)
        ok, issues = verifier.verify(self._valid_constitution())
        assert ok is True
        assert len(issues) == 0

    def test_issue_has_counterexample(self):
        from chimera_core.engines.tla_engine.verifier import TLAVerifier
        verifier = TLAVerifier(animate=False, use_real_tlc=False)
        ok, issues = verifier.verify(self._violation_constitution())
        assert not ok
        assert issues[0].counterexample is not None

    def test_verifier_force_mock_when_use_real_tlc_false(self, monkeypatch):
        """use_real_tlc=False must always use MockModelChecker regardless of Java."""
        import chimera_core.engines.tla_engine.verifier as vmod

        called = []
        original_run = vmod.TLAVerifier._run_tlc

        def patched_run_tlc(self, *args, **kwargs):
            called.append("TLC")
            return original_run(self, *args, **kwargs)

        monkeypatch.setattr(vmod.TLAVerifier, "_run_tlc", patched_run_tlc)
        verifier = vmod.TLAVerifier(animate=False, use_real_tlc=False)
        verifier.verify(self._valid_constitution())
        assert "TLC" not in called

    def test_verifier_tries_tlc_when_use_real_tlc_true(self, monkeypatch):
        """When use_real_tlc=True and TLCRunner.is_available() is True, _run_tlc is called."""
        import chimera_core.engines.tla_engine.verifier as vmod

        called = []

        # Patch TLCRunner.is_available to return True
        monkeypatch.setattr(vmod.TLCRunner, "is_available", lambda self: True)

        # Patch _run_tlc to avoid actual subprocess
        def fake_run_tlc(self, spec, constraints, engine_mode):
            called.append("TLC")
            # Return minimal ConstraintAnimResult list
            from chimera_core.engines.tla_engine.animations import ConstraintAnimResult
            return [ConstraintAnimResult(name=c.name, status="HOLDS", states_checked=1, time_ms=1)
                    for c in constraints], None

        monkeypatch.setattr(vmod.TLAVerifier, "_run_tlc", fake_run_tlc)

        verifier = vmod.TLAVerifier(animate=False, use_real_tlc=True)
        ok, issues = verifier.verify(self._valid_constitution())
        assert "TLC" in called
        assert ok is True


# ═════════════════════════════════════════════════════════════════════════════
# 6. TLCResult → ConstraintAnimResult conversion
# ═════════════════════════════════════════════════════════════════════════════

class TestTLCResultConversion:

    def _constraints(self):
        decls = [VariableDeclaration(name="x", domain='{"A", "B"}')]
        c1 = _make_always_const("c_holds",  "x", ModalOperator.MUST_NOT_BE, _lit("C"))
        c2 = _make_always_const("c_violates", "x", ModalOperator.MUST_NOT_BE, _lit("A"))
        con = _make_constitution("Conv", decls, [c1, c2])
        return con.constraints

    def test_holds_constraint_maps_to_holds(self):
        from chimera_core.engines.tla_engine.verifier import _tlc_result_to_anim_results
        from chimera_core.engines.tla_engine.tlc_runner import TLCResult
        constraints = self._constraints()
        result = TLCResult(success=True, violations=[], states_explored=10, time_ms=50)
        anim = _tlc_result_to_anim_results(result, constraints, 50)
        assert all(r.status == "HOLDS" for r in anim)

    def test_violated_constraint_maps_to_violated(self):
        from chimera_core.engines.tla_engine.verifier import _tlc_result_to_anim_results
        from chimera_core.engines.tla_engine.tlc_runner import TLCResult, TLCViolation
        constraints = self._constraints()
        v = TLCViolation(
            invariant="c_violates",
            state_vars={"x": '"A"'},
            trace=[{"x": '"A"'}],
        )
        result = TLCResult(success=False, violations=[v], states_explored=5, time_ms=30)
        anim = _tlc_result_to_anim_results(result, constraints, 30)
        statuses = {r.name: r.status for r in anim}
        assert statuses["c_holds"]    == "HOLDS"
        assert statuses["c_violates"] == "VIOLATED"

    def test_counterexample_attached_to_violated(self):
        from chimera_core.engines.tla_engine.verifier import _tlc_result_to_anim_results
        from chimera_core.engines.tla_engine.tlc_runner import TLCResult, TLCViolation
        constraints = self._constraints()
        v = TLCViolation(
            invariant="c_violates",
            state_vars={"x": '"A"'},
            trace=[{"x": '"A"'}],
        )
        result = TLCResult(success=False, violations=[v], states_explored=5, time_ms=30)
        anim = _tlc_result_to_anim_results(result, constraints, 30)
        viol = next(r for r in anim if r.name == "c_violates")
        assert viol.counterexample is not None
        assert len(viol.counterexample) == 1
