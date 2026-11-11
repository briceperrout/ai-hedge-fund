"""Robustesse du signal (référence aléatoire, découpage annuel) et portefeuille témoin du paper-trading."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import dashboard
import run_robustness as rb
from data_pipeline import fetch_history

from .conftest import make_macro


# --- Référence aléatoire ----------------------------------------------------
def _pool(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.004, 0.05, n)
    return pd.DataFrame({"year": rng.choice([2021, 2022, 2023], n), "ret": ret, "win": (ret > 0).astype(float)})


def test_signals_drawn_from_the_pool_show_no_edge():
    pool = _pool()
    edge = rb.signal_edge(pool.sample(200, random_state=1), pool, draws=2000)
    assert edge["conclusive"] and 0.05 < edge["p_value"] < 0.95
    assert edge["random_p05"] < edge["signal_avg_ret"] < edge["random_p95"]


def test_a_real_edge_is_detected():
    pool = _pool()
    winners = pool[pool["ret"] > 0.01].sample(200, random_state=1)
    edge = rb.signal_edge(winners, pool, draws=2000)
    assert edge["p_value"] < 0.01 and edge["signal_avg_ret"] > edge["random_p95"]
    assert all(row["signal_win_rate"] > row["base_win_rate"] for row in edge["by_year"])


def test_a_high_win_rate_alone_is_not_an_edge():
    """Objectif proche, stop lointain : 70 % de trades gagnants au hasard, rendement moyen nul."""
    rng = np.random.default_rng(0)
    win = rng.random(6000) < 0.70
    pool = pd.DataFrame({"year": 2023, "ret": np.where(win, 0.03, -0.07), "win": win.astype(float)})
    edge = rb.signal_edge(pool.sample(150, random_state=2), pool, draws=2000)
    assert edge["signal_win_rate"] > 0.6 and edge["p_value"] > 0.05


def test_too_few_signals_is_inconclusive():
    pool = _pool()
    edge = rb.signal_edge(pool.head(12), pool)
    assert edge["conclusive"] is False and "p_value" not in edge


def _daily_pool(seed: int = 0, days: int = 600, names: int = 20) -> pd.DataFrame:
    """Univers dont le rendement dépend surtout du jour (facteur commun) et un peu de la valeur."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    common = rng.normal(0.005, 0.03, days)
    frame = pd.DataFrame({"date": np.repeat(dates, names), "ret": np.repeat(common, names) + rng.normal(0, 0.03, days * names)})
    frame["common"] = np.repeat(common, names)
    return frame


def test_buying_on_good_days_is_timing_not_stock_picking():
    pool = _daily_pool()
    good_days = pool[pool["common"] > 0.02]
    signals = good_days.groupby("date").sample(3, random_state=1)        # valeurs prises au hasard ces jours-là
    split = rb.timing_vs_selection(signals, pool)
    assert split["universe_ret_on_signal_days"] > split["universe_ret_on_other_days"] + 0.02
    assert split["conclusive"] and abs(split["selection_t"]) < 2.5 and abs(split["selection_excess"]) < 0.01


def test_picking_the_best_stock_of_the_day_is_selection():
    pool = _daily_pool()
    best = pool.loc[pool.groupby("date")["ret"].idxmax()].iloc[::2]      # un jour sur deux, la meilleure valeur
    split = rb.timing_vs_selection(best, pool)
    assert split["selection_t"] > 5 and split["selection_excess"] > 0.03


def test_decomposition_needs_enough_blocks():
    pool = _daily_pool(days=100)
    assert rb.timing_vs_selection(pool.groupby("date").head(1), pool)["conclusive"] is False
    assert rb.timing_vs_selection(pool.head(0), pool)["conclusive"] is False


def test_yearly_performance_chains_years_and_counts_trades_by_exit_date():
    days = pd.bdate_range("2022-01-03", "2023-12-29")
    equity = pd.Series(np.linspace(100_000, 110_000, len(days)), index=days)
    trades = pd.DataFrame({"exit_date": ["2022-03-01", "2023-02-01", "2023-06-01"], "pnl": [100.0, -50.0, 80.0]})
    spy = pd.Series(np.linspace(400, 480, len(days)), index=days)
    first, second = rb.yearly_performance(equity, trades, spy)
    assert (first["year"], first["trades"], second["trades"]) == (2022, 1, 2)
    assert second["win_rate"] == 0.5 and first["max_drawdown"] == 0.0
    assert (1 + first["return"]) * (1 + second["return"]) == pytest.approx(1.10)     # aucun jour perdu entre deux années
    assert first["spy_return"] > 0


# --- Deux portefeuilles au tableau de bord ----------------------------------
def _engine(start: str, growth: float, n: int = 40) -> dict:
    days = pd.bdate_range(start, periods=n)
    return {"initial_capital": 100_000.0, "cash": 100_000.0 * (1 + growth), "positions": {}, "pending": {},
            "last_price": {}, "emergency_exit": False,
            "equity_curve": [[d.strftime("%Y-%m-%d"), 100_000.0 * (1 + growth * i / (n - 1))] for i, d in enumerate(days)]}


@pytest.fixture
def live_dir(tmp_path, monkeypatch):
    db = tmp_path / "market.db"

    def connect():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        con.executescript(fetch_history.SCHEMA)
        return con

    with connect() as con:
        for run_id, pnl in (("live-paper", 120.0), ("live-quant", -80.0), ("live-quant", 40.0)):
            con.execute("INSERT INTO trades (run_id, mode, ticker, entry_date, exit_date, exit_reason, pnl, pnl_pct) "
                        "VALUES (?, 'live', 'AAPL', '2024-01-03', '2024-01-10', 'DURÉE MAX', ?, ?)",
                        (run_id, pnl, pnl / 10_000))
        con.commit()
    monkeypatch.setattr(dashboard, "DATA", tmp_path)
    monkeypatch.setattr(dashboard, "db_connect", connect)
    monkeypatch.setattr(dashboard, "load_macro", lambda: make_macro(n=400, start="2023-06-01"))
    return tmp_path


def test_quant_only_portfolio_is_the_main_one_until_the_committee_runs(live_dir):
    (live_dir / "live_state.json").write_text(json.dumps({"paper_quant": {"last_day": "2024-02-23",
                                                                         "engine": _engine("2024-01-02", 0.02)}}))
    data = dashboard.load_live()
    assert data["label"] == "Quantitatif seul" and data["others"] == []
    assert len(data["trades"]) == 2                                  # seuls les trades du témoin
    page = dashboard.render(data)
    assert "quantitatif seul" in page and "Comité IA contre quantitatif seul" not in page


def test_both_portfolios_are_compared_once_the_committee_runs(live_dir):
    (live_dir / "live_state.json").write_text(json.dumps({
        "paper_quant": {"last_day": "2024-02-23", "engine": _engine("2024-01-02", 0.02)},
        "paper": {"last_day": "2024-02-23", "engine": _engine("2024-01-16", 0.01, n=30)}}))
    data = dashboard.load_live()
    assert data["label"] == "Avec comité IA" and [o["label"] for o in data["others"]] == ["Quantitatif seul"]
    assert len(data["trades"]) == 1 and len(data["others"][0]["trades"]) == 2
    page = dashboard.render(data)
    assert "Comité IA contre quantitatif seul" in page
    assert page.count('class="line s') == 3                          # comité, témoin, S&P 500
    assert "+1,00" not in page and "+1.00%" in page and "+2.00%" in page


def test_chart_tolerates_a_second_portfolio_that_started_later(live_dir):
    main = dashboard._curve(_engine("2024-01-02", 0.02))
    late = {"label": "Quantitatif seul", "curve": dashboard._curve(_engine("2024-02-01", 0.01, n=10))}
    chart = dashboard.equity_chart(main, "Avec comité IA", [late])
    payload = json.loads(chart.split('type="application/json">')[1].split("</script>")[0])
    values = payload["series"][1]["values"]
    assert values[0] is None and values[22] == 100.0 and values[31] == pytest.approx(101.0)
    assert "nan" not in chart.lower()
