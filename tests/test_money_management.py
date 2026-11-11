"""Critère de Kelly et dimensionnement : calculés par Python, jamais par un LLM."""
from __future__ import annotations

import numpy as np
import pytest

from quant_engine.xgboost_model import kelly_fraction, plan_trade


def test_kelly_formula():
    # f* = p - (1 - p) / b
    assert kelly_fraction(0.75, 0.5) == pytest.approx(0.25)
    assert kelly_fraction(0.60, 2.0) == pytest.approx(0.40)


def test_kelly_is_zero_without_edge():
    assert kelly_fraction(0.60, 0.5) == 0.0     # 0.6 - 0.4/0.5 < 0
    assert kelly_fraction(0.50, 1.0) == 0.0     # pile ou face
    assert kelly_fraction(0.90, 0.0) == 0.0     # gain nul
    assert kelly_fraction(0.90, -1.0) == 0.0


def test_kelly_is_bounded_and_monotonic():
    values = [kelly_fraction(p, 0.5) for p in np.linspace(0, 1, 51)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values)


def test_plan_applies_fractional_kelly(cfg):
    plan = plan_trade("X", equity=100_000, cash=100_000, price=100.0, atr=2.0, probability=0.75, cfg=cfg)
    assert plan.kelly_full == pytest.approx(0.25)
    assert plan.shares == 62                       # 0,25 x 0,25 = 6,25 % de 100 000 USD à 100 USD
    assert plan.allocation_pct == pytest.approx(0.062)
    assert plan.stop_distance == pytest.approx(cfg.sl_atr * 2.0)
    assert plan.target_distance == pytest.approx(cfg.tp_atr * 2.0)
    assert plan.levels(101.0) == pytest.approx((101.0 - 6.0, 101.0 + 3.0))


def test_plan_is_capped_by_max_position(cfg):
    plan = plan_trade("X", 100_000, 100_000, 50.0, 1.0, 0.999, cfg)
    assert plan.allocation_pct <= cfg.max_position_pct + 1e-9


def test_plan_never_exceeds_available_cash(cfg):
    plan = plan_trade("X", 100_000, 2_000, 100.0, 2.0, 0.90, cfg)
    assert plan.shares * plan.reference_price <= 2_000


@pytest.mark.parametrize("kwargs", [
    {"probability": 0.60},              # Kelly nul : pas d'avantage
    {"cash": 50.0},                     # pas de quoi acheter une action
    {"atr": float("nan")},              # stop impossible à placer
    {"atr": 0.0},
    {"price": 0.0},
])
def test_no_plan_when_trade_cannot_be_sized(cfg, kwargs):
    base = {"equity": 100_000, "cash": 100_000, "price": 100.0, "atr": 2.0, "probability": 0.75}
    assert plan_trade("X", cfg=cfg, **{**base, **kwargs}) is None
