"""Le Cerveau Gauche : features, entraînement XGBoost, probabilités et money management.

Le modèle répond à une seule question : « si j'achète à l'ouverture de demain, quelle est
la probabilité que l'objectif de gain soit touché avant le stop-loss dans les N séances ? »
Cette probabilité alimente ensuite le critère de Kelly (taille de position).

Usage (test du moteur seul, sans LLM) :
    python -m quant_engine.xgboost_model
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# macOS : PyTorch (embeddings) et XGBoost embarquent chacun leur propre runtime OpenMP ; les faire
# tourner en multi-thread dans le même processus provoque un plantage (segfault). Un seul thread
# OpenMP par bibliothèque supprime le conflit. Doit être fixé avant le chargement de ces bibliothèques.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv

from data_pipeline.fetch_history import TICKERS, load_dark_pool, load_macro, load_prices
from quant_engine.filing_features import FILING_FEATURES, load_filing_features

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger("quant")

MODEL_DIR = ROOT / "data" / "models"
BEST_PARAMS_PATH = ROOT / "data" / "best_params.json"

TECH_FEATURES = [
    "rsi_14", "dist_sma20", "dist_sma50", "dist_sma200", "sma50_over_200",
    "ret_1d", "ret_5d", "ret_20d", "volatility_10d", "volatility_60d", "vol_regime",
    "atr_pct", "volume_ratio", "macd_hist", "bollinger_b", "dist_52w_high",
]
MACRO_FEATURES = [
    "vix", "vix_chg_5d", "vix_z_252", "vix_term", "skew", "vvix", "tnx", "tnx_chg_20d",
    "spy_dist_sma200", "spy_ret_5d", "spy_ret_20d", "rel_strength_20d",
]
OPTIONAL_FEATURES = ["dark_short_ratio", "dark_short_z_20"]  # Phase 3, si les données sont en base

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 10, "gamma": 0.5, "reg_alpha": 0.1, "reg_lambda": 2.0,
}


# ---------------------------------------------------------------------------
# Configuration de la stratégie
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    threshold: float = 0.70          # probabilité minimale pour réveiller les agents
    horizon: int = 10                # durée maximale d'un trade (séances)
    tp_atr: float = 1.5              # objectif de gain, en multiples d'ATR
    sl_atr: float = 3.0              # stop-loss, en multiples d'ATR
    kelly_multiplier: float = 0.25   # Kelly fractionné (prudence face à l'erreur de modèle)
    max_position_pct: float = 0.20   # plafond par position
    max_open_positions: int = 5
    cost_bps: float = 5.0            # frais + slippage par exécution

    @property
    def payoff_ratio(self) -> float:
        return self.tp_atr / self.sl_atr

    @classmethod
    def from_env(cls, **overrides: Any) -> "StrategyConfig":
        env = {
            "threshold": float(os.getenv("SIGNAL_THRESHOLD", cls.threshold)),
            "horizon": int(os.getenv("HORIZON_DAYS", cls.horizon)),
            "tp_atr": float(os.getenv("TP_ATR_MULT", cls.tp_atr)),
            "sl_atr": float(os.getenv("SL_ATR_MULT", cls.sl_atr)),
            "kelly_multiplier": float(os.getenv("KELLY_MULTIPLIER", cls.kelly_multiplier)),
            "max_position_pct": float(os.getenv("MAX_POSITION_PCT", cls.max_position_pct)),
            "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", cls.max_open_positions)),
            "cost_bps": float(os.getenv("COST_BPS", cls.cost_bps)),
        }
        env.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env)


# ---------------------------------------------------------------------------
# 1. Ingénierie des caractéristiques (aucune ne regarde le futur)
# ---------------------------------------------------------------------------
def compute_atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = prices["close"].shift(1)
    true_range = pd.concat([
        prices["high"] - prices["low"],
        (prices["high"] - prev_close).abs(),
        (prices["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs macro : peur (VIX), options (SKEW, VVIX, structure par terme), taux, marché."""
    m = macro.ffill()
    out = pd.DataFrame(index=m.index)
    nan = pd.Series(np.nan, index=m.index)
    vix, spy, tnx = m.get("vix", nan), m.get("spy", nan), m.get("tnx", nan)
    out["vix"] = vix
    out["vix_chg_5d"] = vix.diff(5)
    out["vix_z_252"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    out["vix_term"] = vix / m.get("vix3m", nan)
    out["skew"] = m.get("skew", nan)
    out["vvix"] = m.get("vvix", nan)
    out["tnx"] = tnx
    out["tnx_chg_20d"] = tnx.diff(20)
    out["spy_dist_sma200"] = spy / spy.rolling(200).mean() - 1
    out["spy_ret_5d"] = spy.pct_change(5)
    out["spy_ret_20d"] = spy.pct_change(20)
    return out


def build_features(prices: pd.DataFrame, macro_feats: pd.DataFrame,
                   dark_pool: Optional[pd.Series] = None) -> pd.DataFrame:
    """Features d'un ticker. Ligne t = information connue à la clôture de t."""
    close, volume = prices["close"], prices["volume"]
    f = pd.DataFrame(index=prices.index)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f.loc[loss == 0, "rsi_14"] = 100.0

    sma = {w: close.rolling(w).mean() for w in (20, 50, 200)}
    for w in (20, 50, 200):
        f[f"dist_sma{w}"] = close / sma[w] - 1
    f["sma50_over_200"] = sma[50] / sma[200] - 1

    ret = close.pct_change()
    f["ret_1d"], f["ret_5d"], f["ret_20d"] = ret, close.pct_change(5), close.pct_change(20)
    f["volatility_10d"] = ret.rolling(10).std()
    f["volatility_60d"] = ret.rolling(60).std()
    f["vol_regime"] = f["volatility_10d"] / f["volatility_60d"]
    f["atr"] = compute_atr(prices)
    f["atr_pct"] = f["atr"] / close
    f["volume_ratio"] = volume / volume.rolling(20).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    std20 = close.rolling(20).std()
    f["bollinger_b"] = (close - (sma[20] - 2 * std20)) / (4 * std20)
    f["dist_52w_high"] = close / close.rolling(252).max() - 1

    f = f.join(macro_feats.reindex(f.index, method="ffill"))
    f["rel_strength_20d"] = f["ret_20d"] - f["spy_ret_20d"]

    if dark_pool is not None and len(dark_pool):
        ratio = dark_pool.reindex(f.index)
        f["dark_short_ratio"] = ratio
        f["dark_short_z_20"] = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
    return f.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 2. Cible : méthode des barrières (objectif / stop / durée max)
# ---------------------------------------------------------------------------
def label_trades(prices: pd.DataFrame, atr: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    """Simule, pour chaque jour t, le trade qui serait pris à l'ouverture de t+1.

    target = 1 si l'objectif (+tp_atr x ATR) est touché avant le stop (-sl_atr x ATR)
    dans ``horizon`` séances. ``label_end`` est la date à laquelle l'issue est connue :
    un trade n'est utilisable à l'entraînement qu'une fois cette date passée (purge).
    """
    o, h, l, c = (prices[col].to_numpy(float) for col in ("open", "high", "low", "close"))
    a = atr.to_numpy(float)
    dates = prices.index
    n = len(prices)
    target = np.full(n, np.nan)
    trade_ret = np.full(n, np.nan)
    label_end = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    for t in range(n - 1):
        if np.isnan(a[t]) or a[t] <= 0:
            continue
        entry_bar = t + 1
        entry = o[entry_bar]
        take_profit, stop = entry + cfg.tp_atr * a[t], entry - cfg.sl_atr * a[t]
        last_bar = min(entry_bar + cfg.horizon - 1, n - 1)
        for k in range(entry_bar, last_bar + 1):
            if l[k] <= stop:  # stop testé en premier : hypothèse la plus défavorable
                target[t], trade_ret[t], label_end[t] = 0.0, min(o[k], stop) / entry - 1, dates[k]
                break
            if h[k] >= take_profit:
                target[t], trade_ret[t], label_end[t] = 1.0, max(o[k], take_profit) / entry - 1, dates[k]
                break
        else:
            if last_bar == entry_bar + cfg.horizon - 1:  # durée max atteinte : sortie à la clôture
                target[t], trade_ret[t], label_end[t] = 0.0, c[last_bar] / entry - 1, dates[last_bar]
    return pd.DataFrame({"target": target, "trade_ret": trade_ret, "label_end": label_end}, index=dates)


def build_dataset(tickers: Optional[Iterable[str]] = None, cfg: Optional[StrategyConfig] = None,
                  start: Optional[str] = None, end: Optional[str] = None, filing_features: Optional[bool] = None
                  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Assemble features + cibles de tout l'univers.

    ``filing_features`` ajoute les variables extraites des dépôts SEC par le LLM (défaut : variable
    d'environnement ENABLE_FILING_FEATURES, désactivée tant que leur intérêt n'est pas démontré).
    Retourne (dataset indexé par date avec colonne ``ticker``, prix par ticker, liste des features).
    """
    cfg = cfg or StrategyConfig.from_env()
    if filing_features is None:
        filing_features = os.getenv("ENABLE_FILING_FEATURES", "false").lower() == "true"
    filing_signals = None
    if filing_features:
        from llm_agents.filing_extractor import load_signals

        filing_signals = load_signals()
    macro_feats = macro_features(load_macro(end=end))
    frames, prices_by_ticker = [], {}
    for ticker in (tickers or TICKERS):
        prices = load_prices(ticker, end=end)
        if len(prices) < 300:
            log.warning("%s : historique insuffisant (%d séances), ignoré.", ticker, len(prices))
            continue
        feats = build_features(prices, macro_feats, load_dark_pool(ticker))
        if filing_signals is not None:
            feats = feats.join(load_filing_features(ticker, feats.index, filing_signals))
        frame = feats.join(label_trades(prices, feats["atr"], cfg)).join(prices[["open", "high", "low", "close"]])
        frame.insert(0, "ticker", ticker)
        frames.append(frame)
        prices_by_ticker[ticker] = prices
    if not frames:
        raise RuntimeError("Aucune donnée de prix en base : lancez `python -m data_pipeline.fetch_history`.")
    dataset = pd.concat(frames).sort_index(kind="stable")
    if start:
        dataset = dataset.loc[dataset.index >= pd.Timestamp(start)]

    features = TECH_FEATURES + MACRO_FEATURES
    for col in OPTIONAL_FEATURES:  # n'activer une feature alternative que si elle est réellement couverte
        if col in dataset and dataset[col].notna().mean() >= 0.5:
            features.append(col)
    if filing_signals is not None:
        features += [col for col in FILING_FEATURES if dataset[col].notna().any()]
    dataset = dataset.dropna(subset=TECH_FEATURES)
    return dataset, prices_by_ticker, features


def training_rows(dataset: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Lignes dont l'issue était connue à ``cutoff`` (purge anti-fuite de la cible)."""
    known = dataset["target"].notna() & (dataset["label_end"] <= pd.Timestamp(cutoff))
    return dataset.loc[known]


# ---------------------------------------------------------------------------
# 3. Entraînement et prédiction
# ---------------------------------------------------------------------------
def load_best_params() -> dict[str, Any]:
    """Hyperparamètres figés par Optuna (``quant_engine/optimizer.py``), sinon valeurs par défaut."""
    if BEST_PARAMS_PATH.exists():
        return json.loads(BEST_PARAMS_PATH.read_text())["params"]
    return dict(DEFAULT_PARAMS)


def train_model(train_df: pd.DataFrame, features: list[str], params: Optional[dict[str, Any]] = None,
                seed: int = 42) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **(params or load_best_params()),
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=int(os.getenv("XGB_N_JOBS", "4")),
    )
    model.fit(train_df[features], train_df["target"].astype(int))
    return model


def predict_proba(model: xgb.XGBClassifier, features: list[str], rows: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(rows[features])[:, 1]


def get_statistical_signal(model: xgb.XGBClassifier, features: list[str], row: pd.DataFrame,
                           threshold: float) -> dict[str, Any]:
    """Signal du jour pour un ticker : BUY si la probabilité dépasse le seuil, sinon HOLD."""
    probability = float(predict_proba(model, features, row)[0])
    return {"statistical_action": "BUY" if probability >= threshold else "HOLD",
            "win_probability": round(probability, 4)}


def save_model(model: xgb.XGBClassifier, features: list[str], meta: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "xgb_model.json")
    (MODEL_DIR / "xgb_meta.json").write_text(json.dumps({"features": features, **meta}, indent=2, default=str))


def load_model() -> tuple[Optional[xgb.XGBClassifier], dict[str, Any]]:
    path = MODEL_DIR / "xgb_model.json"
    if not path.exists():
        return None, {}
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model, json.loads((MODEL_DIR / "xgb_meta.json").read_text())


# ---------------------------------------------------------------------------
# 4. Money management : critère de Kelly et plan d'exécution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TradePlan:
    """Ordre calculé par Python. Le LLM n'y écrit jamais rien."""
    ticker: str
    shares: int
    reference_price: float
    stop_distance: float      # distance du stop-loss sous le prix d'entrée (USD)
    target_distance: float    # distance de l'objectif au-dessus du prix d'entrée (USD)
    probability: float
    kelly_full: float         # fraction de Kelly théorique
    allocation_pct: float     # fraction du capital réellement engagée

    def levels(self, fill_price: float) -> tuple[float, float]:
        """(stop-loss, objectif) recalés sur le prix d'exécution réel."""
        return fill_price - self.stop_distance, fill_price + self.target_distance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def kelly_fraction(probability: float, payoff_ratio: float) -> float:
    """Critère de Kelly : f* = p - (1 - p) / b, borné à [0, 1]."""
    if payoff_ratio <= 0:
        return 0.0
    return float(min(max(probability - (1.0 - probability) / payoff_ratio, 0.0), 1.0))


def plan_trade(ticker: str, equity: float, cash: float, price: float, atr: float, probability: float,
               cfg: StrategyConfig) -> Optional[TradePlan]:
    """Dimensionne la position (Kelly fractionné, plafonné) et place stop-loss et objectif."""
    full = kelly_fraction(probability, cfg.payoff_ratio)
    allocation = min(cfg.kelly_multiplier * full, cfg.max_position_pct)
    budget = min(allocation * equity, cash * 0.995)  # marge pour frais et écart d'ouverture
    shares = int(budget // price) if price > 0 else 0
    if allocation <= 0 or shares <= 0 or not atr or np.isnan(atr):
        return None
    return TradePlan(
        ticker=ticker, shares=shares, reference_price=float(price),
        stop_distance=float(cfg.sl_atr * atr), target_distance=float(cfg.tp_atr * atr),
        probability=float(probability), kelly_full=full, allocation_pct=shares * price / equity,
    )


# ---------------------------------------------------------------------------
# Test du moteur quantitatif seul
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    from data_pipeline.fetch_history import sync_all

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    config = StrategyConfig.from_env()
    oos_start = pd.Timestamp(os.getenv("OOS_START", "2023-01-01"))
    oos_end = pd.Timestamp(os.getenv("OOS_END", "2023-12-31"))
    sync_all(end=str(oos_end.date()))
    data, _, feature_list = build_dataset(cfg=config, end=str(oos_end.date()))

    train = training_rows(data, oos_start - pd.Timedelta(days=1))
    test = data[(data.index >= oos_start) & data["target"].notna()]
    clf = train_model(train, feature_list)
    proba = predict_proba(clf, feature_list, test)
    picked = test[proba >= config.threshold]

    print(f"\nEntraînement : {len(train)} trades simulés jusqu'au {train.index.max().date()} "
          f"(taux de base {train['target'].mean():.1%})")
    print(f"Test hors-échantillon {oos_start.date()} -> {oos_end.date()} : {len(test)} lignes, "
          f"AUC = {roc_auc_score(test['target'], proba):.3f}")
    print(f"Signaux >= {config.threshold:.0%} : {len(picked)}"
          + (f" | réussite réelle {picked['target'].mean():.1%} | rendement moyen {picked['trade_ret'].mean():+.2%}"
             if len(picked) else ""))
    example = plan_trade("EXEMPLE", 100_000, 100_000, 150.0, 3.0, 0.75, config)
    print(f"Kelly (p=75%, b={config.payoff_ratio:.1f}) : f*={example.kelly_full:.0%} -> "
          f"{example.allocation_pct:.1%} du capital, {example.shares} actions, "
          f"stop -{example.stop_distance:.2f} USD, objectif +{example.target_distance:.2f} USD")
