"""Aucune information future : ni dans les features, ni dans les cibles utilisées à l'entraînement."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.xgboost_model import build_features, label_trades, macro_features, training_rows

from .conftest import flat_prices, make_macro, make_prices


# --- Features ---------------------------------------------------------------
def test_features_do_not_change_when_the_future_is_removed():
    prices, macro = make_prices(), make_macro()
    cut = 300
    full = build_features(prices, macro_features(macro))
    truncated = build_features(prices.iloc[:cut], macro_features(macro.iloc[:cut]))
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_exact=False, rtol=1e-9)


def test_features_do_not_change_when_the_future_is_rewritten():
    prices, macro = make_prices(), make_macro()
    cut = 300
    altered_prices, altered_macro = prices.copy(), macro.copy()
    altered_prices.iloc[cut:] *= 3.0
    altered_macro.iloc[cut:] *= 0.2
    before = build_features(prices, macro_features(macro)).iloc[:cut]
    after = build_features(altered_prices, macro_features(altered_macro)).iloc[:cut]
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-9)


# --- Cibles (barrières) -----------------------------------------------------
def _labels(prices: pd.DataFrame, cfg, atr_value: float = 2.0) -> pd.DataFrame:
    return label_trades(prices, pd.Series(atr_value, index=prices.index), cfg)


def test_target_hit(cfg):
    prices = flat_prices()
    prices.iloc[5, prices.columns.get_loc("high")] = 104.0     # objectif = 100 + 1,5 x 2 = 103
    row = _labels(prices, cfg).iloc[2]                          # signal en t=2, entrée à l'ouverture de t=3
    assert row["target"] == 1.0
    assert row["trade_ret"] == pytest.approx(0.03)
    assert row["label_end"] == prices.index[5]


def test_stop_hit(cfg):
    prices = flat_prices()
    prices.iloc[5, prices.columns.get_loc("low")] = 93.0       # stop = 100 - 3 x 2 = 94
    row = _labels(prices, cfg).iloc[2]
    assert row["target"] == 0.0
    assert row["trade_ret"] == pytest.approx(-0.06)


def test_stop_wins_when_both_barriers_are_touched_the_same_day(cfg):
    prices = flat_prices()
    prices.iloc[5, prices.columns.get_loc("high")] = 110.0
    prices.iloc[5, prices.columns.get_loc("low")] = 90.0
    assert _labels(prices, cfg).iloc[2]["target"] == 0.0        # hypothèse la plus défavorable


def test_gap_down_exits_at_the_open_not_at_the_stop(cfg):
    prices = flat_prices()
    prices.iloc[5, [prices.columns.get_loc(c) for c in ("open", "low", "close", "high")]] = 80.0
    assert _labels(prices, cfg).iloc[2]["trade_ret"] == pytest.approx(-0.20)


def test_timeout_exits_at_the_close_after_horizon_sessions(cfg):
    prices = flat_prices()
    prices.iloc[12, prices.columns.get_loc("close")] = 101.0   # dernière séance du trade : t+1 ... t+10
    row = _labels(prices, cfg).iloc[2]
    assert row["target"] == 0.0
    assert row["trade_ret"] == pytest.approx(0.01)
    assert row["label_end"] == prices.index[12]


def test_outcome_is_unknown_until_the_trade_is_over(cfg):
    labels = _labels(flat_prices(n=40), cfg)
    assert labels["target"].iloc[:30].notna().all()
    assert labels["target"].iloc[30:].isna().all()              # horizon incomplet : pas de cible


def test_label_is_always_known_strictly_after_the_signal(cfg):
    prices = make_prices()
    labels = label_trades(prices, pd.Series(prices["close"] * 0.02, index=prices.index), cfg)
    known = labels.dropna(subset=["target"])
    assert len(known) > 300
    assert (known["label_end"] > known.index).all()


def test_labels_before_a_date_ignore_prices_after_it(cfg):
    prices = make_prices()
    atr = pd.Series(prices["close"] * 0.02, index=prices.index)
    cut = 300
    full = label_trades(prices, atr, cfg)
    truncated = label_trades(prices.iloc[:cut], atr.iloc[:cut], cfg)
    usable = truncated["target"].notna()
    pd.testing.assert_frame_equal(full.iloc[:cut][usable], truncated[usable])


# --- Purge de l'échantillon d'entraînement ---------------------------------
def test_training_rows_only_contain_trades_already_closed(cfg):
    prices = make_prices()
    dataset = label_trades(prices, pd.Series(prices["close"] * 0.02, index=prices.index), cfg)
    cutoff = prices.index[200]
    train = training_rows(dataset, cutoff)
    assert len(train) > 100
    assert train["label_end"].max() <= cutoff
    assert train["target"].notna().all()
    # des trades signalés avant la date, mais encore ouverts, sont exclus
    excluded = dataset[(dataset.index <= cutoff) & ~dataset.index.isin(train.index)]
    assert len(excluded) > 0
    assert (excluded["label_end"].isna() | (excluded["label_end"] > cutoff)).all()


def test_model_score_of_a_day_is_unaffected_by_later_rows(cfg):
    """Bout en bout : entraîner à une date puis noter cette date donne le même score avec ou sans le futur."""
    from quant_engine.xgboost_model import TECH_FEATURES, predict_proba, train_model

    prices, macro = make_prices(n=600), make_macro(n=600)
    params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}

    def score(until: int) -> float:
        p, m = prices.iloc[:until], macro.iloc[:until]
        feats = build_features(p, macro_features(m))
        data = feats.join(label_trades(p, feats["atr"], cfg)).dropna(subset=TECH_FEATURES)
        day = prices.index[449]
        model = train_model(training_rows(data, day), TECH_FEATURES, params=params)
        return float(predict_proba(model, TECH_FEATURES, data.loc[[day]])[0])

    assert score(450) == pytest.approx(score(600), abs=1e-7)
    assert not np.isnan(score(450))
