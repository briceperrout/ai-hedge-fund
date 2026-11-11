"""Moteur d'exécution : ordres, stop-loss, objectif, durée max, arrêt d'urgence, comptabilité."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_engine.xgboost_model import TradePlan
from run_backtest import ExecutionEngine, PendingOrder

DAYS = pd.bdate_range("2023-01-02", periods=30)
COST = 5 / 1e4


def bar(open_=100.0, high=None, low=None, close=None, ticker="X") -> dict:
    return {ticker: {"open": open_, "high": high if high is not None else open_,
                     "low": low if low is not None else open_, "close": close if close is not None else open_}}


def plan(ticker="X", shares=100, price=100.0, stop=6.0, target=3.0) -> TradePlan:
    return TradePlan(ticker=ticker, shares=shares, reference_price=price, stop_distance=stop,
                     target_distance=target, probability=0.75, kelly_full=0.25, allocation_pct=0.1)


def open_position(engine: ExecutionEngine, day=DAYS[0], open_=100.0, **plan_kwargs) -> None:
    engine.pending["X"] = PendingOrder(plan(**plan_kwargs), "2022-12-30")
    engine.on_open(day, bar(open_))


def step(engine: ExecutionEngine, day, bars) -> list[dict]:
    closed = engine.on_open(day, bars) + engine.on_session(day, bars)
    engine.mark(day, bars)
    return closed


def test_order_is_filled_at_the_next_open_with_costs(engine):
    open_position(engine, open_=102.0)
    pos = engine.positions["X"]
    assert not engine.pending
    assert pos.entry_price == pytest.approx(102.0 * (1 + COST))
    assert (pos.stop_loss, pos.take_profit) == pytest.approx((96.0, 105.0))   # recalés sur l'ouverture réelle
    assert engine.cash == pytest.approx(100_000 - 100 * 102.0 * (1 + COST))


def test_order_waits_when_the_ticker_does_not_trade(engine):
    engine.pending["X"] = PendingOrder(plan(), "2022-12-30")
    engine.on_open(DAYS[0], bar(ticker="OTHER"))
    assert "X" in engine.pending and not engine.positions


def test_order_size_is_reduced_to_available_cash(engine):
    engine.cash = 1_000.0
    open_position(engine)
    assert engine.positions["X"].shares == 9
    assert engine.cash >= 0


def test_stop_loss(engine):
    open_position(engine)
    (trade,) = step(engine, DAYS[1], bar(100.0, high=101.0, low=93.0, close=95.0))
    assert trade["exit_reason"] == "STOP-LOSS"
    assert trade["exit_price"] == pytest.approx(94.0 * (1 - COST))
    assert trade["pnl"] < 0 and not engine.positions


def test_gap_below_the_stop_exits_at_the_open(engine):
    open_position(engine)
    (trade,) = step(engine, DAYS[1], bar(85.0, high=86.0, low=84.0, close=85.0))
    assert trade["exit_price"] == pytest.approx(85.0 * (1 - COST))


def test_take_profit(engine):
    open_position(engine)
    (trade,) = step(engine, DAYS[1], bar(100.0, high=104.0, low=99.0, close=103.5))
    assert trade["exit_reason"] == "OBJECTIF ATTEINT"
    assert trade["exit_price"] == pytest.approx(103.0 * (1 - COST))
    assert trade["pnl"] > 0


def test_stop_is_tested_before_the_target(engine):
    open_position(engine)
    (trade,) = step(engine, DAYS[1], bar(100.0, high=110.0, low=90.0))
    assert trade["exit_reason"] == "STOP-LOSS"


def test_max_holding_period(engine, cfg):
    open_position(engine)
    engine.on_session(DAYS[0], bar())                       # la séance d'entrée compte pour 1
    for day in DAYS[1:cfg.horizon - 1]:
        assert step(engine, day, bar()) == []
    (trade,) = step(engine, DAYS[cfg.horizon - 1], bar(100.0, close=101.0))
    assert trade["exit_reason"] == "DURÉE MAX"
    assert trade["bars_held"] == cfg.horizon
    assert trade["exit_price"] == pytest.approx(101.0 * (1 - COST))


def test_emergency_exit_closes_everything_at_the_open(engine):
    open_position(engine)
    engine.pending["Y"] = PendingOrder(plan(ticker="Y"), "2023-01-02")
    engine.emergency_exit = True
    closed = engine.on_open(DAYS[1], {**bar(98.0), **bar(50.0, ticker="Y")})
    assert [t["exit_reason"] for t in closed] == ["ARRÊT D'URGENCE (veto systémique)"]
    assert closed[0]["exit_price"] == pytest.approx(98.0 * (1 - COST))
    assert not engine.positions and not engine.pending      # l'ordre en attente est annulé, pas exécuté
    assert engine.emergency_exit is False                   # l'alerte est consommée


def test_accounting_identity(engine, journal):
    """Liquidités finales = capital initial + somme des P&L, coûts inclus."""
    scenario = [bar(100.0), bar(100.0, high=104.0), bar(50.0), bar(50.0, low=40.0), bar(60.0), bar(60.0, high=70.0)]
    for i, bars in enumerate(scenario):
        if not engine.positions and not engine.pending:
            engine.pending["X"] = PendingOrder(plan(price=bars["X"]["open"]), DAYS[i].strftime("%Y-%m-%d"))
        step(engine, DAYS[i], bars)
    engine.liquidate(DAYS[len(scenario)], "FIN DE SIMULATION")
    assert not engine.positions and len(engine.closed) >= 3
    assert engine.cash == pytest.approx(engine.initial_capital + sum(t["pnl"] for t in engine.closed))
    assert engine.equity() == pytest.approx(engine.cash)
    assert len(journal) == len(engine.closed)               # chaque trade clôturé est journalisé
    assert all(t["pnl"] == pytest.approx((t["exit_price"] - t["entry_price"]) * t["shares"]) for t in engine.closed)


def test_equity_marks_open_positions_at_the_close(engine):
    open_position(engine)
    value = engine.mark(DAYS[0], bar(100.0, close=110.0))
    assert value == pytest.approx(engine.cash + 100 * 110.0)
    assert engine.equity_curve[-1] == ("2023-01-02", value)


def test_buying_power_and_slots_account_for_pending_orders(engine, cfg):
    engine.pending["X"] = PendingOrder(plan(shares=100, price=100.0), "2023-01-02")
    assert engine.buying_power() == pytest.approx(90_000)
    assert engine.free_slots() == cfg.max_open_positions - 1


def test_state_survives_a_json_round_trip(engine, cfg):
    open_position(engine)
    engine.pending["Y"] = PendingOrder(plan(ticker="Y"), "2023-01-02", {"news": "dépêche"})
    engine.cooldown["Z"] = "2023-01-09"
    engine.mark(DAYS[0], bar(100.0, close=101.0))
    state = json.loads(json.dumps(engine.to_state(), default=str))
    restored = ExecutionEngine.from_state(state, cfg, "live")
    assert restored.cash == engine.cash
    assert restored.positions == engine.positions
    assert restored.pending["Y"].plan == engine.pending["Y"].plan
    assert restored.pending["Y"].context == {"news": "dépêche"}
    assert restored.cooldown == engine.cooldown
    assert restored.equity() == pytest.approx(engine.equity())
    assert restored.equity_curve == engine.equity_curve
