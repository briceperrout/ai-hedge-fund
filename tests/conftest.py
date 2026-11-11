"""Fixtures communes : aucune ne touche au réseau, aux modèles LLM ni à la base de données réelle."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_backtest  # noqa: E402
from quant_engine.xgboost_model import StrategyConfig  # noqa: E402


@pytest.fixture
def cfg() -> StrategyConfig:
    """Stratégie figée (indépendante du .env) : objectif 1,5 ATR, stop 3 ATR, 10 séances, 5 bps."""
    return StrategyConfig(threshold=0.70, horizon=10, tp_atr=1.5, sl_atr=3.0, kelly_multiplier=0.25,
                          max_position_pct=0.20, max_open_positions=5, cost_bps=5.0)


@pytest.fixture
def journal(monkeypatch) -> list[dict]:
    """Remplace l'écriture SQL des trades clôturés par une liste en mémoire."""
    written: list[dict] = []

    def fake_journal(trade: dict) -> int:
        written.append(trade)
        return len(written)

    monkeypatch.setattr(run_backtest, "_journal_trade", fake_journal)
    return written


@pytest.fixture
def engine(cfg, journal) -> run_backtest.ExecutionEngine:
    return run_backtest.ExecutionEngine(cfg, 100_000.0, "test-run", "backtest")


def make_prices(n: int = 420, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Marche aléatoire OHLCV cohérente (low <= open, close <= high)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def make_macro(n: int = 420, seed: int = 1, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "vix": 20 + np.cumsum(rng.normal(0, 0.5, n)),
        "vix3m": 22 + np.cumsum(rng.normal(0, 0.4, n)),
        "skew": 130 + rng.normal(0, 3, n),
        "vvix": 100 + rng.normal(0, 5, n),
        "tnx": 3 + np.cumsum(rng.normal(0, 0.02, n)),
        "spy": 400 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
    }, index=index)


def flat_prices(n: int = 40, price: float = 100.0, start: str = "2023-01-02") -> pd.DataFrame:
    """Cours parfaitement plat : sert de base aux scénarios écrits à la main."""
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1e6}, index=index)
