"""Statistiques de l'expérience comparative et rendu du tableau de bord (sans réseau ni base réelle)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dashboard
import run_experiment as xp

from .conftest import make_macro, make_prices


# --- Bras témoin ------------------------------------------------------------
def test_random_veto_is_reproducible_and_close_to_its_rate():
    def draws(seed):
        committee = xp.random_veto_committee(0.3, seed)
        return [committee({}, {})["decision"].veto_activated for _ in range(2000)]

    assert draws(1) == draws(1)
    assert draws(1) != draws(2)
    assert np.mean(draws(1)) == pytest.approx(0.3, abs=0.03)


def test_random_veto_never_triggers_the_emergency_exit():
    committee = xp.random_veto_committee(1.0, 0)
    decision = committee({}, {})["decision"]
    assert decision.veto_activated and not decision.systemic_risk_detected


def test_veto_rate_counts_only_signals_seen_by_the_committee():
    decisions = pd.DataFrame({"outcome": ["APPROVED", "APPROVED", "VETO", "ERROR", "NO_SLOT", "NO_SIZE"]})
    assert xp.veto_rate(decisions) == (pytest.approx(0.4), 2, 5)
    assert xp.veto_rate(pd.DataFrame()) == (0.0, 0, 0)


# --- Test de permutation ----------------------------------------------------
def test_permutation_test_detects_a_real_difference():
    rng = np.random.default_rng(0)
    diff, p = xp.permutation_test(rng.normal(0.03, 0.02, 40), rng.normal(-0.03, 0.02, 40), draws=2000)
    assert diff > 0.04 and p < 0.01


def test_permutation_test_stays_quiet_on_noise():
    rng = np.random.default_rng(0)
    _, p = xp.permutation_test(rng.normal(0, 0.02, 40), rng.normal(0, 0.02, 40), draws=2000)
    assert p > 0.05


def test_signal_analysis_refuses_to_conclude_on_small_samples():
    decisions = pd.DataFrame({"outcome": ["APPROVED"] * 10 + ["VETO"] * 2, "hypothetical_ret": [0.01] * 12})
    result = xp.signal_level_analysis(decisions)
    assert result["conclusive"] is False and result["blocked"] == 2


def test_signal_analysis_ignores_signals_never_submitted_or_still_open():
    decisions = pd.DataFrame({
        "outcome": ["APPROVED"] * 6 + ["VETO"] * 6 + ["NO_SLOT"] * 3 + ["VETO"],
        "hypothetical_ret": [0.05] * 6 + [-0.05] * 6 + [0.5] * 3 + [np.nan]})
    result = xp.signal_level_analysis(decisions)
    assert (result["approved"], result["blocked"]) == (6, 6)
    assert result["avg_ret_difference"] == pytest.approx(0.10)
    assert result["p_value"] < 0.01


def test_cro_is_ranked_inside_the_random_distribution():
    runs = [{"total_return": r, "sharpe": s, "max_drawdown": -0.01, "trades": 10}
            for r, s in zip(np.linspace(0, 0.02, 21), np.linspace(0, 2, 21))]
    cro = {"total_return": 0.02, "sharpe": 1.0, "max_drawdown": -0.01, "trades": 10}
    summary = xp.summarize_random_arm(runs, cro)
    assert summary["draws"] == 21
    assert summary["sharpe"]["cro_percentile"] == pytest.approx(11 / 21)
    assert summary["total_return"]["cro_percentile"] == 1.0
    assert "cro_percentile" not in xp.summarize_random_arm(runs, None)["sharpe"]


def test_verdict_never_claims_value_without_evidence():
    runs = [{"total_return": 0.01, "sharpe": s, "max_drawdown": -0.01, "trades": 10} for s in (0.5, 1.0, 1.5)]
    cro = {"total_return": 0.01, "sharpe": 2.0, "max_drawdown": -0.01, "trades": 10}
    summary = xp.summarize_random_arm(runs, cro)
    text = xp.verdict(cro, summary, {"conclusive": False, "reason": "trop peu de vetos"})
    assert "non concluant" in text and "discrimine mieux" not in text
    assert "aucune conclusion" in xp.verdict(None, xp.summarize_random_arm(runs, None), None)


# --- Tableau de bord --------------------------------------------------------
@pytest.fixture
def live_data(monkeypatch):
    prices, macro = make_prices(n=120), make_macro(n=120)
    monkeypatch.setattr(dashboard, "load_prices", lambda ticker: prices)
    monkeypatch.setattr(dashboard, "load_macro", lambda: macro)
    days = prices.index
    curve = pd.Series(100_000 * (1 + np.linspace(0, 0.03, 60)), index=days[40:100])
    return {
        "title": "Paper-trading en continu", "subtitle": "test", "initial": 100_000.0, "cash": 80_000.0,
        "curve": curve, "emergency": True, "last_price": {"AAPL": 110.0},
        "positions": {"AAPL": {"shares": 50, "entry_price": 100.0, "stop_loss": 94.0, "take_profit": 103.0,
                               "entry_date": "2020-04-01"}},
        "pending": {"MSFT": {"plan": {"shares": 20}}},
        "trades": pd.DataFrame([{"ticker": "NVDA", "entry_date": "2020-03-02", "exit_date": "2020-03-09",
                                 "exit_reason": "STOP-LOSS", "pnl": -420.0, "pnl_pct": -0.06}]),
        "decisions": pd.DataFrame([
            {"date": days[45].strftime("%Y-%m-%d"), "ticker": "NVDA", "probability": 0.74, "outcome": "VETO",
             "justification": "VIX en <backwardation>"},
            {"date": days[50].strftime("%Y-%m-%d"), "ticker": "AAPL", "probability": 0.71, "outcome": "APPROVED",
             "justification": "Rien à signaler dans le dossier."},
            {"date": days[119].strftime("%Y-%m-%d"), "ticker": "MSFT", "probability": 0.72, "outcome": "VETO",
             "justification": "Résultats trimestriels demain."},
        ]),
    }


def test_blocked_signals_get_their_outcome_once_known(live_data):
    decisions = dashboard.add_hypothetical_outcomes(live_data["decisions"])
    assert decisions["hypothetical_ret"].iloc[:2].notna().all()
    assert pd.isna(decisions["hypothetical_ret"].iloc[2])       # trade encore ouvert : pas d'issue inventée
    board = dashboard.veto_scoreboard(decisions)
    assert (board["signals"], board["blocked"], board["blocked_known"]) == (3, 2, 1)


def test_kpis(live_data):
    kpis = dashboard.compute_kpis(live_data)
    assert kpis["total_return"] == pytest.approx(0.03)
    assert kpis["max_drawdown"] == pytest.approx(0.0)
    assert kpis["win_rate"] == 0.0 and kpis["trades"] == 1


def test_page_renders_every_section_and_escapes_llm_text(live_data):
    page = dashboard.render(live_data)
    for expected in ("Paper-trading en continu", "<svg", "S&amp;P 500", "Alerte systémique", "AAPL", "MSFT ×20",
                     "STOP-LOSS", "en cours", "Le comité a étudié <b>3</b> signaux"):
        assert expected in page
    assert "&lt;backwardation&gt;" in page and "<backwardation>" not in page   # texte du LLM jamais injecté tel quel


def test_page_renders_before_the_first_cycle():
    empty = {"title": "t", "subtitle": "s", "initial": 0.0, "cash": 0.0, "curve": pd.Series(dtype=float),
             "positions": {}, "pending": {}, "last_price": {}, "emergency": False, "trades": pd.DataFrame(),
             "decisions": pd.DataFrame()}
    page = dashboard.render(empty)
    assert "aucune séance traitée" in page and "Aucune position ouverte." in page
