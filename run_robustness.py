"""Test de robustesse du signal quantitatif : plus d'années, plus de valeurs, et une vraie référence.

Le backtest de 2023 sur dix valeurs repose sur une vingtaine de trades. Ce script pose trois questions :

    1. Le résultat tient-il année après année, de 2019 à aujourd'hui ?
    2. Tient-il sur des valeurs que l'optimisation des hyperparamètres n'a jamais vues ?
    3. Les signaux font-ils mieux qu'un tirage au hasard parmi les mêmes valeurs, les mêmes années ?

La troisième question est la plus importante : avec un objectif à 1,5 ATR et un stop à 3 ATR, la
majorité des trades sont gagnants quel que soit le jour d'entrée. Un taux de réussite élevé ne prouve
donc rien tant qu'il n'est pas comparé au taux de base.

Aucun LLM n'est utilisé : tout signal est exécuté.

Usage :
    python run_robustness.py
    python run_robustness.py --start 2019-01-01 --universes core new
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.fetch_history import TICKERS, sync_all
from quant_engine.xgboost_model import StrategyConfig, build_dataset
from run_backtest import REPORT_DIR, run_backtest

# Quarante grandes valeurs américaines de tous secteurs, gagnantes et perdantes de la période,
# absentes de l'univers sur lequel Optuna a choisi les hyperparamètres.
NEW_UNIVERSE = [
    "INTC", "AMD", "QCOM", "TXN", "CSCO", "IBM", "ORCL", "ADBE", "CRM", "NFLX",
    "DIS", "T", "VZ", "PFE", "MRK", "JNJ", "ABBV", "LLY", "CVS", "KO",
    "PEP", "PG", "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "BA", "CAT",
    "DE", "GE", "HON", "UPS", "CVX", "COP", "BAC", "C", "GS", "V",
]
UNIVERSES = {"core": ("Univers d'origine (10 valeurs)", TICKERS), "new": ("Valeurs jamais vues (40)", NEW_UNIVERSE)}
TUNING_END_YEAR = 2022      # Optuna a vu les données jusqu'à fin 2022 (univers d'origine uniquement)
DRAWS = 10_000


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------
def yearly_performance(equity: pd.Series, trades: pd.DataFrame, spy: pd.Series) -> list[dict[str, Any]]:
    """Rendement, Sharpe, perte maximale et trades de chaque année civile."""
    rows = []
    previous_close = None
    for year, curve in equity.groupby(equity.index.year):
        base = previous_close if previous_close is not None else curve.iloc[0]
        returns = pd.concat([pd.Series([base]), curve]).pct_change().dropna()
        closed = trades[pd.to_datetime(trades["exit_date"]).dt.year == year] if len(trades) else trades
        market = spy[spy.index.year == year].dropna()
        rows.append({
            "year": int(year), "sessions": int(len(curve)),
            "return": float(curve.iloc[-1] / base - 1),
            "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0,
            "max_drawdown": float((curve / curve.cummax() - 1).min()),
            "trades": int(len(closed)),
            "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else None,
            "spy_return": float(market.iloc[-1] / market.iloc[0] - 1) if len(market) > 1 else None,
        })
        previous_close = curve.iloc[-1]
    return rows


def signal_edge(signals: pd.DataFrame, pool: pd.DataFrame, draws: int = DRAWS, seed: int = 0) -> dict[str, Any]:
    """Les signaux font-ils mieux que des entrées tirées au hasard, année par année ?

    ``signals`` et ``pool`` portent les colonnes ``year``, ``ret`` (rendement du trade simulé) et ``win``.
    Le tirage témoin prend, chaque année, autant d'entrées au hasard qu'il y a eu de signaux.
    """
    by_year = []
    for year, group in signals.groupby("year"):
        base = pool[pool["year"] == year]
        if not len(base):
            continue
        by_year.append({"year": int(year), "signals": int(len(group)),
                        "signal_win_rate": float(group["win"].mean()), "base_win_rate": float(base["win"].mean()),
                        "signal_avg_ret": float(group["ret"].mean()), "base_avg_ret": float(base["ret"].mean())})
    out: dict[str, Any] = {"by_year": by_year, "signals": int(len(signals))}
    if len(signals) < 30:
        return {**out, "conclusive": False, "reason": "moins de 30 signaux"}
    rng = np.random.default_rng(seed)
    pools = {int(y): g["ret"].to_numpy() for y, g in pool.groupby("year")}
    counts = [(int(y), len(g)) for y, g in signals.groupby("year") if int(y) in pools]
    simulated = np.zeros(draws)
    for year, n in counts:
        simulated += rng.choice(pools[year], size=(draws, n)).sum(axis=1)
    simulated /= sum(n for _, n in counts)
    observed = float(signals["ret"].mean())
    return {**out, "conclusive": True,
            "signal_avg_ret": observed, "signal_win_rate": float(signals["win"].mean()),
            "random_avg_ret": float(simulated.mean()),
            "random_p05": float(np.percentile(simulated, 5)), "random_p95": float(np.percentile(simulated, 95)),
            "p_value": float(((simulated >= observed).sum() + 1) / (draws + 1))}


def timing_vs_selection(signals: pd.DataFrame, pool: pd.DataFrame, block: int = 10) -> dict[str, Any]:
    """Sépare deux sources possibles d'avantage, en tenant compte du regroupement des signaux dans le temps.

    - calendrier : les jours à signal sont-ils de meilleurs jours pour acheter n'importe quelle valeur de l'univers ?
    - sélection  : ces jours-là, la valeur désignée fait-elle mieux que la moyenne de l'univers ?

    Les trades durent jusqu'à dix séances et se chevauchent : les jours sont regroupés en blocs de ``block``
    séances, et le t de Student est calculé sur les moyennes de blocs, pas sur les signaux individuels.
    ``signals`` et ``pool`` portent les colonnes ``date`` et ``ret``.
    """
    daily = pool.groupby("date")["ret"].mean()
    if not len(signals) or not len(daily):
        return {"conclusive": False, "reason": "aucun signal"}
    excess = (signals["ret"] - signals["date"].map(daily)).groupby(signals["date"]).mean().dropna()
    calendar = pd.Series(daily.index.isin(excess.index), index=daily.index)
    blocks = pd.Series(np.arange(len(daily)) // block, index=daily.index)
    selection = excess.groupby(blocks.reindex(excess.index)).mean()
    out: dict[str, Any] = {
        "signal_days": int(len(excess)), "all_days": int(len(daily)), "blocks": int(len(selection)),
        "universe_ret_on_signal_days": float(daily[calendar].mean()),
        "universe_ret_on_other_days": float(daily[~calendar].mean()) if (~calendar).any() else None,
        "selection_excess": float(selection.mean()),
    }
    if len(selection) < 20:
        return {**out, "conclusive": False, "reason": "moins de 20 blocs de dix séances avec signal"}
    t_stat = float(selection.mean() / (selection.std(ddof=1) / np.sqrt(len(selection))))
    return {**out, "conclusive": True, "selection_t": t_stat}


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def run_universe(name: str, tickers: list[str], start: str, end: str, capital: float, retrain_days: int,
                 reuse: str | None = None) -> dict[str, Any]:
    cfg = StrategyConfig.from_env()
    if reuse:  # backtest déjà calculé : on ne refait que l'analyse
        folder = REPORT_DIR / reuse
        report = json.loads((folder / "report.json").read_text())
        tickers, end = report["tickers"], report["period"]["end"]
    else:
        sync_all(tickers, end=end)
        report = run_backtest(start, end, tickers, capital=capital, use_llm=False, use_memory=False, autopsy=False,
                              retrain_days=retrain_days, with_dossier=False, sync=False, show_report=False, tag=name)
        folder = Path(report["report_dir"])
    equity = pd.read_csv(folder / "equity.csv", parse_dates=["date"]).set_index("date")["equity"]
    trades = pd.read_csv(folder / "trades.csv") if (folder / "trades.csv").exists() else pd.DataFrame()
    decisions = pd.read_csv(folder / "decisions.csv")

    dataset, _, _ = build_dataset(tickers, cfg, end=end)
    pool = dataset[(dataset.index >= pd.Timestamp(start)) & dataset["target"].notna()]
    pool = pd.DataFrame({"date": pool.index, "year": pool.index.year, "ret": pool["trade_ret"].to_numpy(),
                         "win": pool["target"].to_numpy()})
    known = decisions[decisions["hypothetical_ret"].notna()]     # tous les signaux, exécutés ou non (faute de place)
    signals = pd.DataFrame({"date": pd.to_datetime(known["date"]), "year": pd.to_datetime(known["date"]).dt.year,
                            "ret": known["hypothetical_ret"],
                            "win": known["hypothetical_win"]})

    from data_pipeline.fetch_history import load_macro

    return {
        "universe": name, "tickers": report["tickers"], "run_id": report["run_id"],
        "total_return": report["total_return"], "sharpe": report["sharpe"], "max_drawdown": report["max_drawdown"],
        "trades": report["trades"], "benchmarks": report["benchmarks"],
        "yearly": yearly_performance(equity, trades, load_macro(end=end)["spy"]),
        "edge": signal_edge(signals, pool),
        "decomposition": timing_vs_selection(signals, pool),
    }


def render_markdown(result: dict[str, Any]) -> str:
    pct = lambda v, d=1: "n/d" if v is None else f"{v:+.{d}%}"  # noqa: E731
    lines = [f"# Robustesse du signal quantitatif ({result['start']} → {result['end']})", "",
             "Moteur quantitatif seul, tout signal exécuté. Les hyperparamètres ont été choisis par Optuna sur "
             f"l'univers d'origine, avec des données allant jusqu'à fin {TUNING_END_YEAR} : ces années-là ne sont "
             "donc pas un test indépendant pour cet univers (marquées *).", ""]
    for name, block in result["universes"].items():
        lines += [f"## {UNIVERSES[name][0]}", "",
                  f"Sur toute la période : rendement {pct(block['total_return'])}, Sharpe {block['sharpe']:.2f}, "
                  f"perte maximale {pct(block['max_drawdown'])}, {block['trades']} trades. "
                  f"S&P 500 en achat-conservation : {pct(block['benchmarks'].get('spy_buy_and_hold'))}.", "",
                  "| Année | Rendement | Sharpe | Perte max | Trades | Gagnants | S&P 500 |", "|---|---|---|---|---|---|---|"]
        for row in block["yearly"]:
            star = "*" if name == "core" and row["year"] <= TUNING_END_YEAR else ""
            win = f"{row['win_rate']:.0%}" if row["win_rate"] is not None else "n/d"
            lines.append(f"| {row['year']}{star} | {pct(row['return'], 2)} | {row['sharpe']:.2f} | "
                         f"{pct(row['max_drawdown'], 2)} | {row['trades']} | {win} | {pct(row['spy_return'])} |")
        edge = block["edge"]
        lines += ["", "Signaux contre entrées au hasard (mêmes valeurs, mêmes années) :", "",
                  "| Année | Signaux | Réussite des signaux | Réussite de base | Rendement moyen des signaux | Rendement moyen de base |",
                  "|---|---|---|---|---|---|"]
        for row in edge["by_year"]:
            lines.append(f"| {row['year']} | {row['signals']} | {row['signal_win_rate']:.0%} | {row['base_win_rate']:.0%} | "
                         f"{pct(row['signal_avg_ret'], 2)} | {pct(row['base_avg_ret'], 2)} |")
        lines.append("")
        if edge.get("conclusive"):
            lines.append(f"Sur {edge['signals']} signaux, rendement moyen par trade {pct(edge['signal_avg_ret'], 2)} contre "
                         f"{pct(edge['random_avg_ret'], 2)} pour des entrées au hasard (intervalle 5 %–95 % : "
                         f"{pct(edge['random_p05'], 2)} à {pct(edge['random_p95'], 2)}). p = {edge['p_value']:.3f}.")
        else:
            lines.append(f"Comparaison au hasard non concluante : {edge['reason']}.")
        split = block.get("decomposition", {})
        if "selection_excess" in split:
            lines += ["", "D'où vient l'écart ?", "",
                      f"- Calendrier : les {split['signal_days']} jours à signal, acheter n'importe quelle valeur de "
                      f"l'univers rapportait {pct(split['universe_ret_on_signal_days'], 2)} par trade, contre "
                      f"{pct(split['universe_ret_on_other_days'], 2)} les autres jours.",
                      f"- Sélection : ces jours-là, l'écart entre la valeur désignée et la moyenne de l'univers est de "
                      f"{pct(split['selection_excess'], 2)}"
                      + (f" (t = {split['selection_t']:.2f} sur {split['blocks']} blocs de dix séances ; "
                         "il faut dépasser 2 environ pour parler d'un effet)." if split.get("conclusive")
                         else f" ({split['reason']}).")]
        lines.append("")
    lines += ["## Lecture", "",
              "- La p-valeur est indulgente : des signaux émis à quelques jours d'écart sur la même valeur ne sont "
              "pas indépendants, donc l'incertitude réelle est plus large que celle affichée.",
              "- L'univers est choisi aujourd'hui (biais du survivant) et l'exécution est idéalisée.", ""]
    return "\n".join(lines)


def run_robustness(start: str, end: str, universes: list[str], capital: float, retrain_days: int,
                   reuse: dict[str, str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"study_id": f"robust-{datetime.now():%Y%m%d-%H%M%S}", "start": start, "end": end,
                              "universes": {}}
    for name in universes:
        label, tickers = UNIVERSES[name]
        print(f"[{label}] backtest {start} -> {end}...", flush=True)
        result["universes"][name] = run_universe(name, list(tickers), start, end, capital, retrain_days,
                                                 (reuse or {}).get(name))
    out_dir = REPORT_DIR / result["study_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    (out_dir / "summary.md").write_text(render_markdown(result))
    print("\n" + render_markdown(result))
    print(f"Rapport : {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustesse du signal quantitatif dans le temps et sur d'autres valeurs.")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--universes", nargs="*", default=list(UNIVERSES), choices=list(UNIVERSES))
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--retrain-days", type=int, default=21)
    parser.add_argument("--reuse", nargs="*", default=[], metavar="UNIVERS=RUN_ID",
                        help="Réutiliser des backtests déjà calculés, ex. core=bt-...-core new=bt-...-new")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    run_robustness(args.start, args.end, args.universes, args.capital, args.retrain_days,
                   dict(item.split("=", 1) for item in args.reuse))
