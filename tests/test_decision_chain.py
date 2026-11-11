"""Chaîne de décision : XGBoost propose, le comité ne peut que bloquer, Python dimensionne.

Les modèles sont remplacés par des doublures : ces tests vérifient la logique de contrôle,
pas la qualité des réponses d'un LLM.
"""
from __future__ import annotations

import pandas as pd
import pytest

import run_backtest
from llm_agents import graph
from llm_agents.schemas import VetoDecision
from quant_engine.xgboost_model import plan_trade

DAY = pd.Timestamp("2023-03-10")


def verdict(veto: bool = False, systemic: bool = False, error: str | None = None) -> dict:
    decision = VetoDecision(
        veto_activated=veto, final_action="HOLD" if veto else "BUY", systemic_risk_detected=systemic,
        risk_level="HIGH" if veto else "LOW", confidence_score=0.9, key_risks=["risque"],
        justification="Justification factuelle tirée du dossier.")
    return {"decision": decision, "bull_thesis": "bull", "bear_thesis": "bear", "error": error}


@pytest.fixture
def scan(cfg, monkeypatch):
    """Lance ``evaluate_signals`` sur un univers factice dont les probabilités sont imposées."""
    monkeypatch.setattr(run_backtest, "predict_proba", lambda model, features, rows: rows["p"].to_numpy())

    def run(probabilities: dict[str, float], committee=None, *, held=(), cooldown=None, slots=5, cash=100_000.0):
        rows = pd.DataFrame({
            "ticker": list(probabilities), "p": list(probabilities.values()), "close": 100.0, "atr": 2.0,
            "trade_ret": 0.01, "target": 1.0}, index=[DAY] * len(probabilities))
        decisions: list[dict] = []
        cooldown = {} if cooldown is None else cooldown
        approved, systemic = run_backtest.evaluate_signals(
            DAY, rows, None, [], cfg, prices={}, macro=pd.DataFrame(), held=set(held), cooldown=cooldown,
            equity=100_000.0, cash=cash, slots=slots, mode="backtest", scope="test", use_llm=False,
            use_memory=False, decisions=decisions, committee=committee, with_dossier=False)
        return approved, systemic, decisions, cooldown

    return run


def test_nothing_happens_below_the_threshold(scan):
    calls = []
    approved, systemic, decisions, _ = scan({"A": 0.69, "B": 0.10}, lambda s, d: calls.append(s) or verdict())
    assert approved == [] and decisions == [] and not systemic
    assert calls == []                                       # le comité n'est même pas réveillé


def test_committee_only_sees_buy_signals_from_the_model(scan):
    seen = []
    scan({"A": 0.80}, lambda signal, dossier: seen.append((signal, dossier["ticker"])) or verdict())
    assert seen == [({"statistical_action": "BUY", "win_probability": 0.80}, "A")]


def test_approval_leads_to_an_order_sized_by_python(scan, cfg):
    approved, _, decisions, _ = scan({"A": 0.80}, lambda s, d: verdict())
    expected = plan_trade("A", 100_000.0, 100_000.0, 100.0, 2.0, 0.80, cfg)
    assert [o.plan for o in approved] == [expected]          # la taille ne dépend d'aucun champ du verdict
    assert decisions[0]["outcome"] == "APPROVED"


def test_veto_blocks_the_order_and_starts_a_cooldown(scan):
    approved, systemic, decisions, cooldown = scan({"A": 0.80}, lambda s, d: verdict(veto=True))
    assert approved == [] and not systemic
    assert decisions[0]["outcome"] == "VETO"
    assert cooldown["A"] == "2023-03-17"                     # 5 jours ouvrés


def test_ticker_in_cooldown_or_already_held_is_skipped(scan):
    approved, _, decisions, _ = scan({"A": 0.90, "B": 0.85, "C": 0.80}, lambda s, d: verdict(),
                                     held={"A"}, cooldown={"B": "2023-03-10"})
    assert [o.plan.ticker for o in approved] == ["C"]
    assert [d["ticker"] for d in decisions] == ["C"]


def test_systemic_veto_raises_the_alarm_and_stops_the_scan(scan):
    calls = []

    def committee(signal, dossier):
        calls.append(dossier["ticker"])
        return verdict(veto=True, systemic=True)

    approved, systemic, decisions, _ = scan({"A": 0.90, "B": 0.80}, committee)
    assert systemic is True and approved == []
    assert calls == ["A"]                                    # plus aucun achat étudié ce soir-là
    assert decisions[0]["systemic"] is True


def test_committee_failure_blocks_the_trade(scan):
    failing = lambda s, d: {**verdict(), "decision": VetoDecision.fail_safe("comité indisponible."),  # noqa: E731
                            "error": "ConnectError"}
    approved, systemic, decisions, _ = scan({"A": 0.80}, failing)
    assert approved == [] and not systemic                   # fail-closed, sans liquider le portefeuille
    assert decisions[0]["outcome"] == "ERROR"


def test_signals_are_ranked_and_limited_by_free_slots(scan):
    approved, _, decisions, _ = scan({"A": 0.75, "B": 0.95, "C": 0.85}, lambda s, d: verdict(), slots=2)
    assert [o.plan.ticker for o in approved] == ["B", "C"]
    assert {d["ticker"]: d["outcome"] for d in decisions}["A"] == "NO_SLOT"


def test_orders_of_the_same_evening_share_the_cash(scan):
    approved, _, decisions, _ = scan({"A": 0.99, "B": 0.98, "C": 0.97}, lambda s, d: verdict(), cash=30_000.0)
    assert sum(o.plan.shares * o.plan.reference_price for o in approved) <= 30_000.0
    assert decisions[-1]["outcome"] in {"APPROVED", "NO_SIZE"}


def test_without_committee_every_signal_is_executed(scan):
    approved, _, _, _ = scan({"A": 0.80, "B": 0.75})
    assert len(approved) == 2


# --- Enveloppe du comité LLM (llm_agents.graph) -----------------------------
DOSSIER = {"ticker": "A", "as_of": "2023-03-10", "technical": "", "fundamentals": "", "macro": "", "news": "",
           "experience": ""}


class FakeApp:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result, self.exc = result, exc

    async def ainvoke(self, state):
        if self.exc:
            raise self.exc
        return self.result


def test_run_committee_fails_closed_when_models_are_down(monkeypatch):
    monkeypatch.setattr(graph, "committee_app", FakeApp(exc=ConnectionError("Ollama éteint")))
    out = graph.run_committee({"statistical_action": "BUY", "win_probability": 0.8}, DOSSIER)
    assert out["decision"].veto_activated and not out["decision"].systemic_risk_detected
    assert "ConnectionError" in out["error"]


def test_run_committee_rejects_a_tampered_decision(monkeypatch):
    tampered = {**verdict()["decision"].model_dump(), "final_action": "HOLD"}   # incohérent avec veto=false
    monkeypatch.setattr(graph, "committee_app", FakeApp(result={"decision": tampered}))
    out = graph.run_committee({"statistical_action": "BUY", "win_probability": 0.8}, DOSSIER)
    assert out["decision"].veto_activated and out["error"]


def test_llm_cannot_create_an_order_without_a_statistical_signal(monkeypatch):
    approve = {"decision": verdict()["decision"].model_dump(), "bull_thesis": "b", "bear_thesis": "b"}
    monkeypatch.setattr(graph, "committee_app", FakeApp(result=approve))
    out = graph.run_committee({"statistical_action": "HOLD", "win_probability": 0.4}, DOSSIER)
    assert out["decision"].veto_activated                    # muraille de Chine


def test_extract_json_strips_surrounding_text():
    assert graph._extract_json('Voici ma réponse :\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert graph._extract_json("pas de json") == "pas de json"
