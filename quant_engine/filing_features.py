"""Variables tirées des dépôts SEC, alignées jour par jour sans regarder le futur.

Entrée : une ligne par dépôt lu par le LLM (``llm_agents.filing_extractor.load_signals``).
Sortie : pour chaque séance, ce que disaient les dépôts déjà lisibles à la clôture de ce jour-là.

  - rapports (10-Q, 10-K, 8-K rubrique 2.02) : moyenne des notes des 100 derniers jours, maximum pour les risques ;
  - événements (autres 8-K)                 : nombre, ton cumulé et risque maximal sur 30 jours.

Avant le premier dépôt lu, ou quand plus aucun rapport n'est assez récent, la valeur est manquante (NaN) :
XGBoost sait traiter l'absence d'information, il ne faut pas la déguiser en zéro.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

REPORT_WINDOW_DAYS = 100    # un rapport trimestriel reste « le dernier connu » environ un trimestre
EVENT_WINDOW_DAYS = 30

REPORT_MEAN = {"fil_guidance": "guidance_direction", "fil_demand_tone": "demand_tone",
               "fil_margin_trend": "margin_trend", "fil_rev_growth": "revenue_growth_yoy_pct",
               "fil_shareholder_return": "shareholder_return"}
REPORT_MAX = {"fil_liquidity_risk": "liquidity_risk", "fil_legal_risk": "legal_risk"}
FILING_FEATURES = [*REPORT_MEAN, *REPORT_MAX, "fil_days_since_report", "fil_events_30d", "fil_event_tone_30d",
                   "fil_event_risk_30d"]


def is_report(signals: pd.DataFrame) -> pd.Series:
    """Rapport périodique ou communiqué de résultats (8-K comportant la rubrique 2.02).

    Le tri repose sur les métadonnées de la SEC, pas sur l'étiquette choisie par le LLM, qui peut se tromper.
    """
    items = signals["items"].fillna("").astype(str) if "items" in signals else pd.Series("", index=signals.index)
    return signals["form"].isin(["10-Q", "10-K"]) | (signals["form"].eq("8-K") & items.str.contains("2.02", regex=False))


def features_from_signals(signals: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features d'une valeur. ``signals`` : dépôts de cette valeur avec leur colonne ``available``."""
    out = pd.DataFrame(np.nan, index=index, columns=FILING_FEATURES)
    if not len(signals) or not len(index):
        return out
    signals = signals.copy()
    # Première séance à la clôture de laquelle le dépôt est connu (jamais avant sa date de disponibilité)
    signals["pos"] = index.searchsorted(pd.to_datetime(signals["available"]).to_numpy(), side="left")
    signals = signals[signals["pos"] < len(index)].sort_values("pos")
    if not len(signals):
        return out
    days = index.to_numpy()

    def window(pos: int, length: int) -> slice:
        stop = int(np.searchsorted(days, days[pos] + np.timedelta64(length, "D"), side="left"))
        return slice(pos, stop)

    reports = signals[is_report(signals)]
    for feature, column in {**REPORT_MEAN, **REPORT_MAX}.items():
        total, count = np.zeros(len(index)), np.zeros(len(index))
        peak = np.full(len(index), -np.inf)
        for pos, value in zip(reports["pos"], pd.to_numeric(reports[column], errors="coerce")):
            if pd.isna(value):
                continue
            span = window(int(pos), REPORT_WINDOW_DAYS)
            total[span] += value
            count[span] += 1
            peak[span] = np.maximum(peak[span], value)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = total / count if feature in REPORT_MEAN else np.where(np.isfinite(peak), peak, np.nan)
        out[feature] = np.where(count > 0, values, np.nan)

    last_report = np.full(len(index), np.datetime64("NaT"), dtype="datetime64[ns]")
    for pos in reports["pos"]:
        last_report[int(pos):] = days[int(pos)]
    age = (days - last_report) / np.timedelta64(1, "D")
    out["fil_days_since_report"] = np.where(age <= REPORT_WINDOW_DAYS, age, np.nan)

    events = signals[~is_report(signals)]
    covered = np.arange(len(index)) >= int(signals["pos"].min())   # avant le premier dépôt lu : inconnu, pas zéro
    count, tone, risk = np.zeros(len(index)), np.zeros(len(index)), np.zeros(len(index))
    for pos, demand, legal in zip(events["pos"], pd.to_numeric(events["demand_tone"], errors="coerce").fillna(0),
                                  pd.to_numeric(events["legal_risk"], errors="coerce").fillna(0)):
        span = window(int(pos), EVENT_WINDOW_DAYS)
        count[span] += 1
        tone[span] += demand
        risk[span] = np.maximum(risk[span], legal)
    out["fil_events_30d"] = np.where(covered, count, np.nan)
    out["fil_event_tone_30d"] = np.where(covered, tone, np.nan)
    out["fil_event_risk_30d"] = np.where(covered, risk, np.nan)
    return out


def load_filing_features(ticker: str, index: pd.DatetimeIndex, signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Features d'une valeur à partir des extractions en base (ou d'un tableau ``signals`` fourni)."""
    if signals is None:
        from llm_agents.filing_extractor import load_signals  # import tardif : charge LangChain

        signals = load_signals()
    if not len(signals):
        return pd.DataFrame(np.nan, index=index, columns=FILING_FEATURES)
    return features_from_signals(signals[signals["ticker"] == ticker.upper()], index)
