"""Expérience contrôlée : le veto du CRO apporte-t-il quelque chose, ou n'importe quel filtre ferait-il pareil ?

Trois bras rejouent la même période avec exactement les mêmes modèles XGBoost (donc les mêmes signaux) :

    A. quantitatif seul   : tout signal est exécuté ;
    B. comité IA          : débat Bull/Bear puis veto du CRO (système complet) ;
    C. veto aléatoire     : chaque signal est bloqué au hasard, au même taux que le CRO (N tirages).

Le bras C est le témoin : bloquer des trades réduit mécaniquement l'exposition et peut améliorer
le Sharpe par pur hasard. Le CRO n'a de valeur que s'il fait mieux que cette distribution.

Deux lectures sont produites :
    - portefeuille : rendement, Sharpe et perte maximale de B, situés dans la distribution de C ;
    - signal par signal : rendement qu'auraient eu les signaux autorisés contre les signaux bloqués,
      avec un test de permutation (indépendant des contraintes de capital et de places).

Usage :
    python run_experiment.py                         # les trois bras (modèles LLM requis)
    python run_experiment.py --cro-run bt-...-cro    # réutilise un backtest CRO déjà calculé
    python run_experiment.py --veto-rate 0.3         # sans LLM : bras A et C seulement (répétition technique)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_pipeline.fetch_history import TICKERS
from llm_agents.schemas import VetoDecision
from run_backtest import REPORT_DIR, Committee, run_backtest

METRICS = ("total_return", "sharpe", "max_drawdown", "trades")
CONSULTED = ("APPROVED", "NO_SIZE", "VETO", "ERROR")   # signaux réellement soumis au comité
BLOCKED = ("VETO", "ERROR")
MIN_GROUP = 5                                          # en dessous, aucune conclusion statistique


# ---------------------------------------------------------------------------
# Bras témoin
# ---------------------------------------------------------------------------
def random_veto_committee(rate: float, seed: int) -> Committee:
    """Comité factice : bloque chaque signal avec la probabilité ``rate``, sans rien lire du dossier."""
    rng = np.random.default_rng(seed)

    def committee(signal: dict[str, Any], dossier: dict[str, Any]) -> dict[str, Any]:
        veto = bool(rng.random() < rate)
        decision = VetoDecision(
            veto_activated=veto, final_action="HOLD" if veto else "BUY", systemic_risk_detected=False,
            risk_level="MEDIUM", confidence_score=0.0, key_risks=[],
            justification=f"Veto aléatoire de contrôle (taux {rate:.0%}, graine {seed}).")
        return {"decision": decision, "bull_thesis": "", "bear_thesis": "", "error": None}

    return committee


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------
def veto_rate(decisions: pd.DataFrame) -> tuple[float, int, int]:
    """(taux de blocage, signaux bloqués, signaux soumis au comité)."""
    if not len(decisions):
        return 0.0, 0, 0
    consulted = decisions[decisions["outcome"].isin(CONSULTED)]
    blocked = int(consulted["outcome"].isin(BLOCKED).sum())
    return (blocked / len(consulted) if len(consulted) else 0.0), blocked, int(len(consulted))


def permutation_test(approved: np.ndarray, blocked: np.ndarray, draws: int = 10_000, seed: int = 0
                     ) -> tuple[float, float]:
    """Écart de rendement moyen (autorisés - bloqués) et p-valeur unilatérale par permutation des étiquettes."""
    observed = float(approved.mean() - blocked.mean())
    pooled, k = np.concatenate([approved, blocked]), len(approved)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        rng.shuffle(pooled)
        hits += pooled[:k].mean() - pooled[k:].mean() >= observed
    return observed, (hits + 1) / (draws + 1)


def signal_level_analysis(decisions: pd.DataFrame) -> dict[str, Any]:
    """Que seraient devenus les signaux autorisés et les signaux bloqués, pris isolément ?"""
    if not len(decisions):
        return {"conclusive": False, "reason": "aucun signal"}
    known = decisions[decisions["outcome"].isin(CONSULTED) & decisions["hypothetical_ret"].notna()]
    approved = known.loc[~known["outcome"].isin(BLOCKED), "hypothetical_ret"].to_numpy(float)
    blocked = known.loc[known["outcome"].isin(BLOCKED), "hypothetical_ret"].to_numpy(float)
    out: dict[str, Any] = {"approved": int(len(approved)), "blocked": int(len(blocked))}
    if len(approved):
        out.update(approved_avg_ret=float(approved.mean()), approved_win_rate=float((approved > 0).mean()))
    if len(blocked):
        out.update(blocked_avg_ret=float(blocked.mean()), blocked_win_rate=float((blocked > 0).mean()))
    if min(len(approved), len(blocked)) < MIN_GROUP:
        return {**out, "conclusive": False,
                "reason": f"moins de {MIN_GROUP} signaux dans un des deux groupes"}
    diff, p_value = permutation_test(approved, blocked)
    return {**out, "conclusive": True, "avg_ret_difference": diff, "p_value": p_value}


def summarize_random_arm(reports: list[dict[str, Any]], cro: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Distribution du bras témoin et rang du CRO dans cette distribution."""
    summary: dict[str, Any] = {"draws": len(reports)}
    for metric in METRICS:
        values = np.array([r[metric] for r in reports], dtype=float)
        stats = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                 "p05": float(np.percentile(values, 5)), "median": float(np.median(values)),
                 "p95": float(np.percentile(values, 95))}
        if cro is not None:  # part des tirages aléatoires que le CRO bat ou égale
            stats["cro_percentile"] = float((values <= cro[metric]).mean())
        summary[metric] = stats
    return summary


def verdict(cro: Optional[dict[str, Any]], random_summary: dict[str, Any], signals: Optional[dict[str, Any]]) -> str:
    if cro is None:
        return "Bras CRO absent (répétition technique) : aucune conclusion sur la valeur du veto."
    lines = []
    rank = random_summary["sharpe"]["cro_percentile"]
    lines.append(f"Portefeuille : le Sharpe du CRO bat ou égale {rank:.0%} des {random_summary['draws']} "
                 "tirages de veto aléatoire au même taux.")
    if signals and signals.get("conclusive"):
        p = signals["p_value"]
        lines.append(f"Signal par signal : les signaux autorisés rapportent {signals['avg_ret_difference']:+.2%} "
                     f"de plus que les signaux bloqués (p = {p:.3f}, test de permutation unilatéral).")
        if rank >= 0.95 and p <= 0.05:
            lines.append("Conclusion : le veto du CRO discrimine mieux que le hasard sur cette période.")
        elif rank <= 0.50 and p >= 0.50:
            lines.append("Conclusion : le veto du CRO ne fait pas mieux qu'un filtre aléatoire sur cette période.")
        else:
            lines.append("Conclusion : résultat non concluant — l'écart au hasard n'est pas établi.")
    else:
        reason = (signals or {}).get("reason", "analyse indisponible")
        lines.append(f"Signal par signal : non concluant ({reason}).")
    lines.append("Réserve : en backtest, le LLM peut connaître la période par son entraînement ; seul le "
                 "paper-trading en temps réel mesure sa valeur sans ce biais.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------
def _fmt(metric: str, value: float) -> str:
    if metric == "sharpe":
        return f"{value:.2f}"
    if metric == "trades":
        return f"{value:.0f}"
    return f"{value:+.2%}"


def render_markdown(result: dict[str, Any]) -> str:
    labels = {"total_return": "Rendement total", "sharpe": "Ratio de Sharpe", "max_drawdown": "Perte maximale",
              "trades": "Trades"}
    quant, cro, rnd = result["quant_only"], result.get("cro"), result["random_veto"]
    header = "| Mesure | A. Quantitatif seul | B. Comité IA | C. Veto aléatoire (médiane, 5 %–95 %) | Rang de B dans C |"
    rows = [header, "|---|---|---|---|---|"]
    for metric in METRICS:
        stats = rnd[metric]
        rows.append(
            f"| {labels[metric]} | {_fmt(metric, quant[metric])} | {_fmt(metric, cro[metric]) if cro else 'n/d'} | "
            f"{_fmt(metric, stats['median'])} ({_fmt(metric, stats['p05'])} à {_fmt(metric, stats['p95'])}) | "
            f"{format(stats['cro_percentile'], '.0%') if 'cro_percentile' in stats else 'n/d'} |")
    period = result["period"]
    lines = [
        f"# Expérience : valeur du veto du CRO ({period['start']} → {period['end']})", "",
        f"Univers : {', '.join(result['tickers'])}. Taux de veto : {result['veto_rate']:.0%} "
        f"({result['veto_rate_source']}). Tirages aléatoires : {rnd['draws']}.", "",
        *rows, "",
    ]
    signals = result.get("signal_level")
    if signals and "approved_avg_ret" in signals and "blocked_avg_ret" in signals:
        lines += [
            "## Signal par signal", "",
            f"- Signaux autorisés : {signals['approved']}, rendement moyen {signals['approved_avg_ret']:+.2%}, "
            f"gagnants {signals['approved_win_rate']:.0%}",
            f"- Signaux bloqués : {signals['blocked']}, rendement moyen qu'ils auraient eu "
            f"{signals['blocked_avg_ret']:+.2%}, gagnants {signals['blocked_win_rate']:.0%}", "",
        ]
    lines += ["## Lecture", "", result["verdict"], ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_cro_run(run_id: str, start: str, end: str, tickers: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    folder = REPORT_DIR / run_id
    if not (folder / "report.json").exists():
        raise SystemExit(f"Backtest introuvable : {folder}")
    report = json.loads((folder / "report.json").read_text())
    if not report.get("llm"):
        raise SystemExit(f"{run_id} n'est pas un backtest avec comité IA.")
    if sorted(report["tickers"]) != sorted(tickers):
        raise SystemExit(f"{run_id} porte sur un autre univers : {report['tickers']}")
    span = report["period"]
    if not (start <= span["start"] and span["end"] <= end):
        raise SystemExit(f"{run_id} couvre {span['start']} -> {span['end']}, pas la période demandée.")
    return report, pd.read_csv(folder / "decisions.csv")


def run_experiment(start: str, end: str, tickers: list[str], *, capital: float, seeds: int, retrain_days: int,
                   fixed_rate: Optional[float] = None, cro_run: Optional[str] = None) -> dict[str, Any]:
    models: dict[pd.Timestamp, Any] = {}    # partagés : les trois bras voient les mêmes probabilités
    common = dict(capital=capital, retrain_days=retrain_days, model_cache=models, show_report=False)

    print(f"Expérience {start} -> {end} sur {len(tickers)} valeurs\n[A] quantitatif seul...", flush=True)
    quant = run_backtest(start, end, tickers, use_llm=False, use_memory=False, autopsy=False,
                         with_dossier=False, tag="quant", **common)

    cro: Optional[dict[str, Any]] = None
    cro_decisions = pd.DataFrame()
    if cro_run:
        cro, cro_decisions = load_cro_run(cro_run, start, end, tickers)
        print(f"[B] comité IA : backtest {cro_run} réutilisé.", flush=True)
    elif fixed_rate is None:
        print("[B] comité IA (débat + veto du CRO) — c'est le bras long...", flush=True)
        cro = run_backtest(start, end, tickers, use_llm=True, use_memory=True, autopsy=True, sync=False,
                           tag="cro", **common)
        cro_decisions = pd.read_csv(Path(cro["report_dir"]) / "decisions.csv")

    if cro is not None:
        rate, blocked, consulted = veto_rate(cro_decisions)
        source = f"observé : {blocked} signaux bloqués sur {consulted}"
    else:
        rate, source = float(fixed_rate), "imposé par --veto-rate"

    print(f"[C] veto aléatoire au taux {rate:.0%} : {seeds} tirages...", flush=True)
    random_reports = []
    for seed in range(seeds):
        random_reports.append(run_backtest(
            start, end, tickers, use_llm=False, use_memory=False, autopsy=False, with_dossier=False, sync=False,
            committee=random_veto_committee(rate, seed), tag=f"rand{seed:03d}", **common))
        print(f"    tirage {seed + 1}/{seeds} : Sharpe {random_reports[-1]['sharpe']:.2f}", flush=True)

    random_summary = summarize_random_arm(random_reports, cro)
    signals = signal_level_analysis(cro_decisions) if cro is not None else None
    keep = lambda r: {m: r[m] for m in METRICS} | {"run_id": r["run_id"]}  # noqa: E731
    result = {
        "experiment_id": f"exp-{datetime.now():%Y%m%d-%H%M%S}",
        "period": quant["period"], "tickers": tickers, "retrain_days": retrain_days,
        "veto_rate": rate, "veto_rate_source": source,
        "quant_only": keep(quant), "cro": keep(cro) if cro else None,
        "random_veto": {**random_summary, "runs": [keep(r) for r in random_reports]},
        "signal_level": signals,
        "benchmarks": quant["benchmarks"],
    }
    result["verdict"] = verdict(result["cro"], random_summary, signals)

    out_dir = REPORT_DIR / result["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    (out_dir / "summary.md").write_text(render_markdown(result))
    print("\n" + render_markdown(result))
    print(f"Rapport : {out_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Expérience contrôlée sur la valeur du veto du CRO.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--capital", type=float, default=float(os.getenv("INITIAL_CAPITAL", "100000")))
    parser.add_argument("--retrain-days", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=50, help="Nombre de tirages du veto aléatoire.")
    parser.add_argument("--cro-run", help="Identifiant d'un backtest avec comité IA déjà calculé (data/reports/).")
    parser.add_argument("--veto-rate", type=float, help="Sans LLM : taux de veto imposé au bras aléatoire.")
    args = parser.parse_args()
    if args.veto_rate is not None and not 0.0 <= args.veto_rate <= 1.0:
        parser.error("--veto-rate doit être compris entre 0 et 1")
    if args.veto_rate is not None and args.cro_run:
        parser.error("--veto-rate et --cro-run sont exclusifs")

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.veto_rate is None and not args.cro_run:
        from llm_agents.graph import check_llm_services

        problems = check_llm_services()
        if problems:
            print("Le comité IA ne peut pas siéger :")
            print("\n".join(f"  - {p}" for p in problems))
            print("Lancez d'abord les modèles, ou utilisez --cro-run / --veto-rate.")
            sys.exit(2)

    run_experiment(args.start, args.end, [t.upper() for t in args.tickers], capital=args.capital,
                   seeds=args.seeds, retrain_days=args.retrain_days, fixed_rate=args.veto_rate,
                   cro_run=args.cro_run)


if __name__ == "__main__":
    main()
