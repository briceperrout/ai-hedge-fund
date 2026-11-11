"""Étude : que vaut un LLM comme source de variables, quand on lui fait lire les dépôts SEC ?

Trois questions, de la plus vérifiable à la plus incertaine :

    1. Coût        : combien d'appels, de jetons et de secondes par document ?
    2. Exactitude  : les chiffres extraits (chiffre d'affaires, BPA) sont-ils ceux des données XBRL officielles ?
                     les citations existent-elles vraiment dans le document ?
    3. Utilité     : ces variables améliorent-elles le classifieur, hors échantillon, davantage que les mêmes
                     variables attribuées à une autre société (témoin) ?

Préalables :
    python -m data_pipeline.fetch_filings          # texte des dépôts
    python -m data_pipeline.fetch_news             # (une fois) chiffres XBRL de référence
    python -m llm_agents.filing_extractor          # lecture par les LLM

Usage :
    python run_filings_study.py
    python run_filings_study.py --start 2023-01-01 --end 2023-12-31 --controls 10
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_pipeline.fetch_history import TICKERS, db_connect

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "data" / "reports"
MIN_FILINGS = 100           # en dessous, aucune conclusion sur l'utilité prédictive
MIN_COVERAGE = 0.5          # part minimale des lignes de test disposant d'au moins une variable
MIN_VERIFIED = 20           # en dessous, le taux d'exactitude est donné à titre indicatif
UNIT_FACTORS = (1e-9, 1e-6, 1e-3, 1e3, 1e6)


# ---------------------------------------------------------------------------
# 1. Coût
# ---------------------------------------------------------------------------
def cost_report(runs: pd.DataFrame) -> dict[str, Any]:
    """``runs`` : une ligne par tentative de lecture, échecs compris."""
    if not len(runs):
        return {"documents": 0}
    ok = runs[runs["error"].isna()]
    report: dict[str, Any] = {
        "documents": int(len(runs)), "failures": int(runs["error"].notna().sum()),
        "models": sorted(runs["model"].unique()),
        "seconds_total": float(runs["seconds"].sum()), "seconds_per_document": float(runs["seconds"].mean()),
        "input_tokens": int(runs["input_tokens"].sum()), "output_tokens": int(runs["output_tokens"].sum()),
        "calls": int(runs["calls"].sum()),
        "by_form": {},
    }
    for form, group in runs.groupby("form"):
        report["by_form"][form] = {
            "documents": int(len(group)), "failures": int(group["error"].notna().sum()),
            "avg_chars": float(group["chars"].mean()), "avg_chunks": float(group["chunks"].mean()),
            "avg_seconds": float(group["seconds"].mean()),
            "avg_tokens": float((group["input_tokens"] + group["output_tokens"]).mean()),
        }
    if len(ok):
        report["grounding"] = float(ok["grounding"].mean())
        report["fully_grounded"] = float((ok["grounding"] == 1.0).mean())
    return report


# ---------------------------------------------------------------------------
# 2. Exactitude (contre les données XBRL officielles)
# ---------------------------------------------------------------------------
def load_reference() -> pd.DataFrame:
    """Chiffres trimestriels officiels (table ``fundamentals``), datés de leur première publication."""
    with db_connect() as con:
        ref = pd.read_sql_query("SELECT ticker, metric, period_start, period_end, value, filed FROM fundamentals", con)
    if not len(ref):
        return ref
    days = (pd.to_datetime(ref["period_end"]) - pd.to_datetime(ref["period_start"])).dt.days
    return ref[days.between(80, 100)].reset_index(drop=True)


def reference_value(reference: pd.DataFrame, ticker: str, metric: str, filed: str) -> Optional[float]:
    """Valeur officielle du trimestre annoncé par un dépôt du ``filed`` (le 10-Q suit le communiqué de 0 à 45 jours)."""
    if not len(reference):
        return None
    filed_ts = pd.Timestamp(filed)
    rows = reference[(reference["ticker"] == ticker) & (reference["metric"] == metric)]
    dates = pd.to_datetime(rows["filed"])
    rows = rows[(dates >= filed_ts - pd.Timedelta(days=1)) & (dates <= filed_ts + pd.Timedelta(days=45))]
    if not len(rows):
        return None
    rows = rows.sort_values(["filed", "period_end"], ascending=[True, False])
    return float(rows.iloc[0]["value"])


def classify(extracted: Optional[float], truth: float, *, absolute: Optional[float] = None) -> str:
    """exact | unit_error (bon chiffre, mauvaise unité) | wrong | missing."""
    if extracted is None or pd.isna(extracted):
        return "missing"
    ratio = extracted / truth if truth else np.inf
    if (abs(extracted - truth) <= absolute) if absolute is not None else (abs(ratio - 1) <= 0.01):
        return "exact"
    if any(abs(ratio / factor - 1) <= 0.01 for factor in UNIT_FACTORS):
        return "unit_error"
    return "wrong"


def accuracy_report(signals: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Compare chiffre d'affaires et BPA extraits aux valeurs officielles, pour les publications de résultats."""
    from quant_engine.filing_features import is_report

    reports = signals[is_report(signals) & signals["form"].ne("10-K")] if len(signals) else signals
    out: dict[str, Any] = {}
    releases = reports[reports["form"].eq("8-K")] if len(reports) else reports
    if len(releases):  # étiquette vérifiable : un 8-K rubrique 2.02 est une publication de résultats
        out["event_type"] = {"verifiable": int(len(releases)), "exact": int(releases["event_type"].eq("EARNINGS").sum()),
                             "exact_rate": float(releases["event_type"].eq("EARNINGS").mean())}
    for name, column, metric, scale, absolute in (("revenue", "revenue_musd", "revenue", 1e6, None),
                                                  ("eps", "eps_diluted", "eps_diluted", 1.0, 0.011)):
        outcomes = []
        for _, row in reports.iterrows():
            truth = reference_value(reference, row["ticker"], metric, row["filed"])
            if truth is None:
                continue
            value = row[column] * scale if pd.notna(row[column]) else None
            outcomes.append(classify(value, truth, absolute=absolute))
        counts = pd.Series(outcomes, dtype=object).value_counts()
        out[name] = {"verifiable": len(outcomes), **{k: int(counts.get(k, 0)) for k in
                                                     ("exact", "unit_error", "wrong", "missing")}}
        if outcomes:
            out[name]["exact_rate"] = counts.get("exact", 0) / len(outcomes)
    return out


# ---------------------------------------------------------------------------
# 3. Utilité prédictive (walk-forward, avec témoin)
# ---------------------------------------------------------------------------
def walk_forward_scores(dataset: pd.DataFrame, features: list[str], start: str, end: str, retrain_days: int
                        ) -> pd.Series:
    """Probabilités hors échantillon : le modèle du jour n'a vu que des trades déjà dénoués."""
    from quant_engine.xgboost_model import predict_proba, train_model, training_rows

    days = [d for d in dataset.index.unique().sort_values() if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    scores = []
    for i in range(0, len(days), retrain_days):
        block = days[i:i + retrain_days]
        train = training_rows(dataset, block[0] - pd.Timedelta(days=1))
        rows = dataset[dataset.index.isin(block)]
        if len(train) < 200 or not len(rows):
            continue
        model = train_model(train, features)
        scores.append(pd.Series(predict_proba(model, features, rows), index=pd.MultiIndex.from_arrays(
            [rows.index, rows["ticker"]])))
    return pd.concat(scores) if scores else pd.Series(dtype=float)


def score_metrics(scores: pd.Series, dataset: pd.DataFrame, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    truth = dataset.set_index([dataset.index, "ticker"])[["target", "trade_ret"]].reindex(scores.index)
    known = truth["target"].notna()
    y, p = truth.loc[known, "target"], scores[known]
    picked = truth.loc[known][p >= threshold]
    out: dict[str, Any] = {"rows": int(known.sum()),
                           "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
                           "signals": int(len(picked))}
    if len(picked):
        out.update(signal_hit_rate=float(picked["target"].mean()), signal_avg_ret=float(picked["trade_ret"].mean()))
    return out


def swap_companies(dataset: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    """Témoin : chaque société reçoit les variables de dépôts d'une AUTRE société, date pour date.

    Calendrier, couverture et distribution des variables sont conservés ; seul le lien avec la société
    est rompu. Si le gain de performance survit à cet échange, il ne venait pas du contenu des dépôts.
    """
    tickers = sorted(dataset["ticker"].unique())
    if len(tickers) < 2:
        raise ValueError("le témoin exige au moins deux sociétés")
    rng = np.random.default_rng(seed)
    while True:
        donors = list(rng.permutation(tickers))
        if all(a != b for a, b in zip(tickers, donors)):
            break
    by_ticker = {t: g[columns] for t, g in dataset.groupby("ticker")}
    swapped = dataset.copy()
    for ticker, donor in zip(tickers, donors):
        mask = swapped["ticker"] == ticker
        swapped.loc[mask, columns] = by_ticker[donor].reindex(swapped.index[mask]).to_numpy()
    return swapped


def predictive_report(tickers: list[str], start: str, end: str, retrain_days: int, controls: int,
                      n_filings: int) -> dict[str, Any]:
    from quant_engine.filing_features import FILING_FEATURES
    from quant_engine.xgboost_model import StrategyConfig, build_dataset

    cfg = StrategyConfig.from_env()
    dataset, _, features = build_dataset(tickers, cfg, end=end, filing_features=True)
    filing = [f for f in features if f in FILING_FEATURES]
    base = [f for f in features if f not in FILING_FEATURES]
    test = dataset[(dataset.index >= pd.Timestamp(start)) & (dataset.index <= pd.Timestamp(end))]
    coverage = float(test[filing].notna().any(axis=1).mean()) if filing and len(test) else 0.0
    report: dict[str, Any] = {"period": {"start": start, "end": end}, "filing_features": filing,
                              "coverage": coverage, "filings_read": n_filings}
    if not filing or dataset["ticker"].nunique() < 2 or not len(test):
        return {**report, "conclusive": False, "reason": "aucune variable de dépôt exploitable sur la période"}

    report["baseline"] = score_metrics(walk_forward_scores(dataset, base, start, end, retrain_days), dataset, cfg.threshold)
    report["with_filings"] = score_metrics(walk_forward_scores(dataset, features, start, end, retrain_days), dataset,
                                           cfg.threshold)
    report["controls"] = [
        score_metrics(walk_forward_scores(swap_companies(dataset, filing, seed), features, start, end, retrain_days),
                      dataset, cfg.threshold) for seed in range(controls)]
    aucs = [report["baseline"]["auc"], report["with_filings"]["auc"], *(c["auc"] for c in report["controls"])]
    if any(a is None for a in aucs):
        return {**report, "conclusive": False, "reason": "AUC incalculable (une seule classe dans la période de test)"}
    report["auc_gain"] = report["with_filings"]["auc"] - report["baseline"]["auc"]
    report["control_gains"] = [c["auc"] - report["baseline"]["auc"] for c in report["controls"]]
    if n_filings < MIN_FILINGS or coverage < MIN_COVERAGE:
        return {**report, "conclusive": False,
                "reason": f"{n_filings} dépôts lus et {coverage:.0%} des lignes de test couvertes "
                          f"(minimum : {MIN_FILINGS} dépôts et {MIN_COVERAGE:.0%})"}
    beats_controls = report["auc_gain"] > max(report["control_gains"])
    report["conclusive"] = True
    report["finding"] = ("gain supérieur à tous les témoins" if beats_controls and report["auc_gain"] > 0.005
                         else "aucun gain distinguable du témoin")
    return report


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------
def render_markdown(result: dict[str, Any]) -> str:
    cost, accuracy, predictive = result["cost"], result["accuracy"], result["predictive"]
    lines = ["# Étude : le LLM comme source de variables (dépôts SEC)", ""]
    if not cost["documents"]:
        return "\n".join(lines + ["Aucun dépôt n'a encore été lu. Lancez `python -m llm_agents.filing_extractor`.", ""])

    lines += [f"Modèles : {', '.join(cost['models'])}. Documents lus : {cost['documents']} "
              f"(dont {cost['failures']} échec(s) d'extraction).", "", "## 1. Coût d'inférence", "",
              "| Formulaire | Documents | Échecs | Caractères lus | Tranches | Secondes | Jetons |", "|---|---|---|---|---|---|---|"]
    for form, row in cost["by_form"].items():
        lines.append(f"| {form} | {row['documents']} | {row['failures']} | {row['avg_chars']:,.0f} | "
                     f"{row['avg_chunks']:.1f} | {row['avg_seconds']:.0f} | {row['avg_tokens']:,.0f} |".replace(",", " "))
    lines += ["", f"Total : {cost['seconds_total'] / 60:.1f} minutes, {cost['calls']} appels, "
                  + f"{cost['input_tokens'] + cost['output_tokens']:,}".replace(",", " ")
                  + " jetons (le tableau donne les moyennes par document).",
              "", "## 2. Exactitude", ""]
    if "grounding" in cost:
        lines.append(f"- Citations retrouvées mot pour mot dans le document : {cost['grounding']:.0%} "
                     f"(documents dont toutes les citations sont exactes : {cost['fully_grounded']:.0%}).")
    labels = {"revenue": "Chiffre d'affaires trimestriel", "eps": "BPA dilué trimestriel"}
    for key, label in labels.items():
        row = accuracy.get(key, {"verifiable": 0})
        if not row["verifiable"]:
            lines.append(f"- {label} : aucune valeur officielle à comparer.")
            continue
        caveat = "" if row["verifiable"] >= MIN_VERIFIED else f" — indicatif, moins de {MIN_VERIFIED} cas"
        lines.append(f"- {label} : {row['exact']}/{row['verifiable']} exacts ({row['exact_rate']:.0%}), "
                     f"{row['unit_error']} erreur(s) d'unité, {row['wrong']} faux, {row['missing']} omis{caveat}.")
    if "event_type" in accuracy:
        row = accuracy["event_type"]
        lines.append(f"- Nature du document reconnue sur les communiqués de résultats : {row['exact']}/{row['verifiable']} "
                     f"({row['exact_rate']:.0%}).")
    lines += ["", "Référence : données XBRL de la SEC, à 1 % près pour le chiffre d'affaires et au centime pour le BPA.",
              "", "## 3. Utilité prédictive", ""]
    if "baseline" in predictive:
        lines += ["| Jeu de variables | AUC hors échantillon | Signaux ≥ seuil | Réussite des signaux |", "|---|---|---|---|"]

        def row(name: str, m: dict[str, Any]) -> str:
            hit = f"{m['signal_hit_rate']:.0%}" if "signal_hit_rate" in m else "n/d"
            auc = f"{m['auc']:.3f}" if m["auc"] is not None else "n/d"
            return f"| {name} | {auc} | {m['signals']} | {hit} |"

        lines += [row("Prix et macro seuls", predictive["baseline"]),
                  row("+ variables des dépôts", predictive["with_filings"])]
        if predictive.get("control_gains"):
            aucs = [c["auc"] for c in predictive["controls"]]
            lines.append(f"| + variables d'une autre société (témoin, {len(aucs)} tirages) | "
                         f"{min(aucs):.3f} à {max(aucs):.3f} | — | — |")
        lines.append("")
    if "auc_gain" in predictive:
        lines.append(f"Gain d'AUC : {predictive['auc_gain']:+.3f} ; gains des témoins : "
                     f"{min(predictive['control_gains']):+.3f} à {max(predictive['control_gains']):+.3f}.")
    lines.append(f"Couverture : {predictive['coverage']:.0%} des lignes de test disposent d'au moins une variable "
                 f"({predictive['filings_read']} dépôts lus).")
    lines.append("")
    lines.append(f"**Conclusion : {predictive['finding']}.**" if predictive.get("conclusive")
                 else f"**Non concluant : {predictive['reason']}.**")
    lines += ["", "Réserve : un LLM entraîné après la période testée peut reconnaître un trimestre et colorer ses notes "
                  "de ce qu'il sait de la suite. Les montants et les citations sont vérifiables ; les notes de ton ne le "
                  "sont pas. Seule une lecture en temps réel, dépôt après dépôt, écarte ce biais.", ""]
    return "\n".join(lines)


def run_study(tickers: list[str], start: str, end: str, retrain_days: int = 21, controls: int = 5) -> dict[str, Any]:
    from llm_agents.filing_extractor import load_signals

    runs = load_signals(include_errors=True)
    runs = runs[runs["ticker"].isin(tickers)] if len(runs) else runs
    signals = runs[runs["error"].isna()] if len(runs) else runs
    result: dict[str, Any] = {"study_id": f"filings-{datetime.now():%Y%m%d-%H%M%S}", "tickers": tickers,
                              "cost": cost_report(runs)}
    result["accuracy"] = accuracy_report(signals, load_reference()) if len(signals) else {}
    result["predictive"] = (predictive_report(tickers, start, end, retrain_days, controls, len(signals)) if len(signals)
                            else {"conclusive": False, "reason": "aucun dépôt lu", "coverage": 0.0, "filings_read": 0})
    out_dir = REPORT_DIR / result["study_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    (out_dir / "summary.md").write_text(render_markdown(result))
    print(render_markdown(result))
    print(f"Rapport : {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue les variables extraites des dépôts SEC par les LLM.")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--retrain-days", type=int, default=21)
    parser.add_argument("--controls", type=int, default=5, help="Nombre de tirages du témoin (sociétés échangées).")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    run_study([t.upper() for t in args.tickers], args.start, args.end, args.retrain_days, args.controls)
