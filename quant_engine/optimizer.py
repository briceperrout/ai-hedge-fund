"""Optimisation bayésienne (Optuna) des hyperparamètres XGBoost.

Objectif : maximiser le ratio de Sharpe hors-échantillon de la stratégie, mesuré en
walk-forward sur la seule période d'optimisation. La période de test finale
(à partir de ``--train-end``) n'est jamais chargée : la séparation Train/Test est stricte.

Les meilleurs hyperparamètres sont figés dans ``data/best_params.json`` et repris
automatiquement par ``xgboost_model.train_model``.

Usage :
    python -m quant_engine.optimizer --trials 60 --train-end 2022-12-31
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd

from data_pipeline.fetch_history import TICKERS, sync_all
from quant_engine.xgboost_model import (
    BEST_PARAMS_PATH, DEFAULT_PARAMS, StrategyConfig, build_dataset, predict_proba, train_model, training_rows,
)

log = logging.getLogger("optimizer")

MIN_TRADES_PER_FOLD = 10


def walk_forward_folds(dataset: pd.DataFrame, n_folds: int, fold_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Découpe la fin de la période d'optimisation en blocs de validation successifs.

    Chaque bloc est prédit par un modèle entraîné uniquement sur les trades dont l'issue
    était connue avant le début du bloc (fenêtre croissante + purge).
    """
    dates = dataset.index.unique().sort_values()
    if len(dates) < (n_folds + 2) * fold_days:
        raise ValueError("Historique trop court pour ce découpage walk-forward.")
    folds = []
    for i in range(n_folds, 0, -1):
        start = dates[len(dates) - i * fold_days]
        end = dates[len(dates) - (i - 1) * fold_days - 1]
        folds.append((start, end))
    return folds


def strategy_sharpe(block: pd.DataFrame, proba: np.ndarray, cfg: StrategyConfig) -> tuple[float, int]:
    """Sharpe annualisé des trades déclenchés (probabilité >= seuil), nets de frais.

    Rendement quotidien = moyenne des trades ouverts ce jour-là (0 les jours sans signal).
    """
    picked = block.loc[proba >= cfg.threshold, "trade_ret"] - 2 * cfg.cost_bps / 1e4
    daily = picked.groupby(level=0).mean().reindex(block.index.unique(), fill_value=0.0)
    std = daily.std()
    sharpe = float(daily.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return sharpe, int(len(picked))


def evaluate_params(params: dict[str, Any], dataset: pd.DataFrame, features: list[str], cfg: StrategyConfig,
                    folds: list[tuple[pd.Timestamp, pd.Timestamp]]) -> tuple[float, list[dict[str, Any]]]:
    scores, details = [], []
    for start, end in folds:
        train = training_rows(dataset, start - pd.Timedelta(days=1))
        block = dataset[(dataset.index >= start) & (dataset.index <= end) & dataset["target"].notna()]
        model = train_model(train, features, params)
        sharpe, n_trades = strategy_sharpe(block, predict_proba(model, features, block), cfg)
        # Trop peu de trades = Sharpe non fiable : on pénalise au lieu de récompenser un coup de chance
        score = sharpe if n_trades >= MIN_TRADES_PER_FOLD else -1.0 + n_trades / MIN_TRADES_PER_FOLD
        scores.append(score)
        details.append({"start": str(start.date()), "end": str(end.date()), "sharpe": round(sharpe, 3),
                        "trades": n_trades})
    # moyenne pénalisée par la dispersion : préférer un Sharpe stable d'un bloc à l'autre
    return float(np.mean(scores) - 0.5 * np.std(scores)), details


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }


def optimize(n_trials: int = 60, train_end: str = "2022-12-31", n_folds: int = 4, fold_days: int = 126,
             tickers: list[str] | None = None, seed: int = 42) -> dict[str, Any]:
    cfg = StrategyConfig.from_env()
    universe = tickers or TICKERS
    sync_all(universe, end=train_end)
    # Séparation stricte : rien après train_end n'entre dans l'optimisation
    dataset, _, features = build_dataset(universe, cfg, end=train_end)
    dataset = training_rows(dataset, pd.Timestamp(train_end))
    folds = walk_forward_folds(dataset, n_folds, fold_days)
    log.info("Optimisation sur %d trades simulés (%s -> %s), %d blocs de validation.",
             len(dataset), dataset.index.min().date(), dataset.index.max().date(), len(folds))

    baseline, _ = evaluate_params(DEFAULT_PARAMS, dataset, features, cfg, folds)

    def objective(trial: optuna.Trial) -> float:
        score, details = evaluate_params(suggest_params(trial), dataset, features, cfg, folds)
        trial.set_user_attr("folds", details)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULT_PARAMS)  # le point de départ connu fait partie de la recherche
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    result = {
        "params": study.best_params,
        "objective": "Sharpe walk-forward moyen pénalisé par sa dispersion",
        "best_score": round(study.best_value, 4),
        "baseline_score": round(baseline, 4),
        "folds": study.best_trial.user_attrs["folds"],
        "n_trials": n_trials,
        "train_end": train_end,
        "tickers": universe,
        "features": features,
        "strategy": {"threshold": cfg.threshold, "horizon": cfg.horizon, "tp_atr": cfg.tp_atr, "sl_atr": cfg.sl_atr},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Fige les hyperparamètres XGBoost par recherche bayésienne.")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--train-end", default="2022-12-31",
                        help="Dernier jour utilisable. Tout ce qui suit est réservé au test hors-échantillon.")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-days", type=int, default=126, help="Taille d'un bloc de validation (séances).")
    args = parser.parse_args()

    best = optimize(args.trials, args.train_end, args.folds, args.fold_days)
    print(f"\nScore par défaut : {best['baseline_score']:+.3f}  ->  score optimisé : {best['best_score']:+.3f}")
    for fold in best["folds"]:
        print(f"  bloc {fold['start']} -> {fold['end']} : Sharpe {fold['sharpe']:+.2f} sur {fold['trades']} trades")
    print(f"Hyperparamètres figés dans {BEST_PARAMS_PATH}")
