"""Génération du dossier d'investissement (recherche Qdrant + macro) à l'instant T.

Tout ce qui entre dans le dossier est strictement antérieur ou égal à ``as_of`` :
prix tronqués, fondamentaux datés de leur publication, dépêches filtrées par horodatage,
leçons du journal de trading clôturées avant la date du signal.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_pipeline import fetch_news
from data_pipeline.fetch_history import db_connect, load_dark_pool, load_options_flow, split_factor
from data_pipeline.qdrant_setup import (
    JOURNAL_COLLECTION, NEWS_COLLECTION, ensure_collections, semantic_search, upsert_documents,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger("snapshot")

NEWS_TOP_K = int(os.getenv("NEWS_TOP_K", "3"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
EXPERIENCE_TOP_K = int(os.getenv("EXPERIENCE_TOP_K", "3"))
MARKET_TZ = "America/New_York"

NO_NEWS = "Aucune dépêche pertinente sur la période."
NO_LESSON = "Aucune leçon comparable dans le journal de trading."


def _day_bounds(as_of: pd.Timestamp) -> tuple[int, int]:
    """Horodatages (début, fin) de la journée de bourse ``as_of`` à New York."""
    start = pd.Timestamp(as_of).normalize().tz_localize(MARKET_TZ)
    return int(start.timestamp()), int((start + pd.Timedelta(days=1)).timestamp()) - 1


# ---------------------------------------------------------------------------
# 1. Contexte technique
# ---------------------------------------------------------------------------
def technical_snapshot(prices: pd.DataFrame) -> str:
    close = prices["close"]
    last = float(close.iloc[-1])
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    prev_close = close.shift(1)
    true_range = pd.concat([prices["high"] - prices["low"], (prices["high"] - prev_close).abs(),
                            (prices["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])

    def dist(window: int) -> Optional[float]:
        return last / float(close.rolling(window).mean().iloc[-1]) - 1 if len(close) >= window else None

    d50, d200 = dist(50), dist(200)
    trend = "indéterminée" if d200 is None else ("haussière" if d200 > 0 else "baissière")
    vol_ratio = float(prices["volume"].iloc[-1] / prices["volume"].rolling(20).mean().iloc[-1])
    ret5 = last / float(close.iloc[-6]) - 1 if len(close) > 5 else float("nan")
    ret20 = last / float(close.iloc[-21]) - 1 if len(close) > 20 else float("nan")
    high_52w = float(close.tail(252).max())
    parts = [
        f"Prix de clôture : {last:.2f} USD.",
        f"RSI(14) : {rsi:.0f}.",
        f"Tendance de fond (SMA200) : {trend}" + (f" ({d200:+.1%} vs SMA200, {d50:+.1%} vs SMA50)." if d200 is not None and d50 is not None else "."),
        f"Performance : {ret5:+.1%} sur 5 séances, {ret20:+.1%} sur 20 séances.",
        f"Distance au plus haut 52 semaines : {last / high_52w - 1:+.1%}.",
        f"Volatilité (ATR14) : {atr:.2f} USD soit {atr / last:.1%} du prix.",
        f"Volume du jour : {vol_ratio:.0%} de la moyenne 20 séances.",
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 2. Contexte macro-économique
# ---------------------------------------------------------------------------
def macro_snapshot(macro: pd.DataFrame) -> str:
    if macro.empty:
        return "Données macro indisponibles."
    m = macro.ffill()
    parts = []
    if "vix" in m:
        vix = float(m["vix"].iloc[-1])
        regime = "calme" if vix < 15 else "normal" if vix < 20 else "tendu" if vix < 30 else "panique"
        chg = vix - float(m["vix"].iloc[-6]) if len(m) > 5 else 0.0
        parts.append(f"VIX (indice de peur) : {vix:.1f}, régime {regime}, {chg:+.1f} pt sur 5 séances.")
        if "vix3m" in m and not np.isnan(m["vix3m"].iloc[-1]):
            ratio = vix / float(m["vix3m"].iloc[-1])
            state = "backwardation = stress immédiat" if ratio > 1 else "contango = marché serein"
            parts.append(f"Structure par terme VIX/VIX3M : {ratio:.2f} ({state}).")
    if "skew" in m and not np.isnan(m["skew"].iloc[-1]):
        parts.append(f"Indice SKEW (couverture de queue via options) : {float(m['skew'].iloc[-1]):.0f}.")
    if "spy" in m:
        spy = m["spy"].dropna()
        if len(spy) >= 200:
            d200 = float(spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1)
            ret5 = float(spy.iloc[-1] / spy.iloc[-6] - 1)
            parts.append(f"S&P 500 (SPY) : {d200:+.1%} vs SMA200, {ret5:+.1%} sur 5 séances.")
    if "tnx" in m and not np.isnan(m["tnx"].iloc[-1]):
        tnx = float(m["tnx"].iloc[-1])
        chg20 = tnx - float(m["tnx"].iloc[-21]) if len(m) > 20 else 0.0
        parts.append(f"Taux 10 ans US : {tnx:.2f}% ({chg20:+.2f} pt sur 20 séances).")
    return " ".join(parts) or "Données macro indisponibles."


# ---------------------------------------------------------------------------
# 3. Fondamentaux « point-in-time » (rapports SEC publiés avant as_of)
# ---------------------------------------------------------------------------
def fundamentals_snapshot(ticker: str, as_of: pd.Timestamp, price: float, mode: str = "backtest") -> str:
    with db_connect() as con:
        df = pd.read_sql_query(
            "SELECT metric, period_start, period_end, value, form, filed FROM fundamentals "
            "WHERE ticker = ? AND filed <= ?", con, params=[ticker, pd.Timestamp(as_of).strftime("%Y-%m-%d")],
        )
    if df.empty:
        return _yfinance_fundamentals(ticker) if mode == "live" else \
            "Fondamentaux indisponibles (aucun rapport SEC publié avant cette date)."
    df["start"], df["end"] = pd.to_datetime(df["period_start"]), pd.to_datetime(df["period_end"])
    df["annual"] = (df["end"] - df["start"]).dt.days > 300
    revenue = df[df["metric"] == "revenue"].sort_values(["end", "filed"])
    if revenue.empty:
        return "Fondamentaux indisponibles (chiffre d'affaires non publié au format XBRL)."
    last = revenue.iloc[-1]
    label = "exercice annuel" if last["annual"] else "trimestre"
    parts = [f"Dernier rapport publié : {last['form']} déposé le {last['filed']} "
             f"({label} clos le {last['period_end']}).",
             f"Chiffre d'affaires : {last['value'] / 1e9:.2f} Md USD"]
    target = last["end"] - pd.Timedelta(days=365)
    prev = revenue[(revenue["annual"] == last["annual"]) & ((revenue["end"] - target).abs() <= pd.Timedelta(days=15))]
    if not prev.empty and prev.iloc[-1]["value"]:
        parts[-1] += f" ({last['value'] / prev.iloc[-1]['value'] - 1:+.1%} sur un an)"
    parts[-1] += "."
    same = df[(df["start"] == last["start"]) & (df["end"] == last["end"])].set_index("metric")["value"]
    if "net_income" in same and last["value"]:
        parts.append(f"Résultat net : {same['net_income'] / 1e9:.2f} Md USD (marge nette {same['net_income'] / last['value']:.1%}).")
    if "eps_diluted" in same:
        parts.append(f"BPA dilué de la période : {same['eps_diluted']:.2f} USD.")
    fy_eps = df[(df["metric"] == "eps_diluted") & df["annual"]].sort_values("end")
    if not fy_eps.empty and fy_eps.iloc[-1]["value"] > 0:
        # Prix en base : ajustés de tous les splits. BPA : dans l'unité de sa date de publication.
        day = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        quoted_price = price * split_factor(ticker, day)                                  # cours coté ce jour-là
        eps_today_units = fy_eps.iloc[-1]["value"] / split_factor(ticker, fy_eps.iloc[-1]["filed"], day)
        parts.append(f"P/E sur le dernier exercice publié : {quoted_price / eps_today_units:.1f}.")
    return " ".join(parts)


def _yfinance_fundamentals(ticker: str) -> str:
    """Repli live uniquement : Yahoo ne fournit que les fondamentaux du jour."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return (f"P/E : {info.get('trailingPE', 'N/A')} (forward {info.get('forwardPE', 'N/A')}). "
                f"Marge nette : {(info.get('profitMargins') or 0):.1%}.")
    except Exception:
        return "Fondamentaux inaccessibles."


# ---------------------------------------------------------------------------
# 4. Données alternatives (Phase 3)
# ---------------------------------------------------------------------------
def alternative_snapshot(ticker: str, as_of: pd.Timestamp) -> str:
    parts = []
    dark = load_dark_pool(ticker).loc[:as_of]
    if len(dark) >= 20:
        parts.append(f"Dark pools (FINRA) : {dark.iloc[-1]:.0%} du volume hors-marché vendu à découvert "
                     f"(moyenne 20 séances : {dark.tail(20).mean():.0%}).")
    options = load_options_flow(ticker).loc[:as_of]
    if not options.empty:
        row = options.iloc[-1]
        parts.append(f"Flux d'options : ratio Put/Call {row['pc_volume_ratio']:.2f} en volume, "
                     f"{row['pc_oi_ratio']:.2f} en positions ouvertes.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 5. RAG d'information (dépêches) et RAG d'expérience (journal de trading)
# ---------------------------------------------------------------------------
def get_contextual_news(ticker: str, as_of: pd.Timestamp, limit: int = NEWS_TOP_K,
                        lookback_days: int = NEWS_LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """Les ``limit`` documents les plus pertinents, jamais publiés après ``as_of``.

    Dépêches : fenêtre courte (``lookback_days``). Dépôts SEC et rapports trimestriels :
    fenêtre de 45 jours, car ils restent pertinents plus longtemps.
    """
    _, ts_max = _day_bounds(as_of)
    query = (f"{fetch_news.company_name(ticker)} {ticker} material risk: lawsuit, investigation, regulation, "
             "earnings, guidance cut, downgrade, insider selling, systemic market stress")
    scope = [ticker, fetch_news.MACRO]
    try:
        hits = semantic_search(NEWS_COLLECTION, query, tickers=scope, kinds=["news"],
                               ts_min=ts_max - lookback_days * 86400, ts_max=ts_max, limit=limit)
        hits += semantic_search(NEWS_COLLECTION, query, tickers=scope,
                                kinds=["sec_filing", "insider", "quarterly_report"],
                                ts_min=ts_max - 45 * 86400, ts_max=ts_max, limit=limit)
    except Exception as exc:
        log.warning("Recherche Qdrant impossible : %s", exc)
        return []
    return sorted(hits, key=lambda h: h["score"], reverse=True)[:limit]


def format_news(hits: list[dict]) -> str:
    if not hits:
        return NO_NEWS
    return "\n".join(f"- [{h.get('date')}] ({h.get('ticker')}, {h.get('source')}) {h.get('text')}" for h in hits)


def get_experience_lessons(context: str, as_of: pd.Timestamp, scope: Optional[str],
                           limit: int = EXPERIENCE_TOP_K) -> list[dict[str, Any]]:
    """Leçons du journal de trading proches du contexte, issues de trades clôturés avant ``as_of``."""
    ts_start, _ = _day_bounds(as_of)
    try:
        return semantic_search(JOURNAL_COLLECTION, context, ts_max=ts_start - 1,
                               kinds=[scope] if scope else None, limit=limit)
    except Exception as exc:
        log.warning("Journal de trading inaccessible : %s", exc)
        return []


def format_lessons(hits: list[dict]) -> str:
    if not hits:
        return NO_LESSON
    return "\n".join(
        f"- [{h.get('exit_date')}] {h.get('ticker')} ({h.get('pnl_pct', 0):+.1%}, {h.get('category')}) : "
        f"{h.get('lesson')} Règle : {h.get('avoid_rule')}" for h in hits
    )


def record_lesson(trade: dict[str, Any], report: dict[str, Any], scope: str) -> None:
    """Stocke l'autopsie d'un trade perdant sous forme de vecteur (RAG d'expérience)."""
    context = trade.get("context", {})
    text = (f"{report['lesson']} Cause : {report['root_cause']} Signaux d'alerte : "
            f"{'; '.join(report.get('warning_signs', []))}. Contexte à l'entrée : "
            f"{context.get('technical', '')} {context.get('macro', '')}")
    _, exit_ts = _day_bounds(pd.Timestamp(trade["exit_date"]))
    upsert_documents(JOURNAL_COLLECTION, [{
        "key": f"lesson:{scope}:{trade['ticker']}:{trade['entry_date']}:{trade['exit_date']}",
        "text": text,
        "kind": scope,
        "ticker": trade["ticker"],
        "entry_date": str(trade["entry_date"])[:10],
        "exit_date": str(trade["exit_date"])[:10],
        "timestamp": exit_ts,
        "pnl_pct": float(trade["pnl_pct"]),
        "category": report["error_category"],
        "root_cause": report["root_cause"],
        "lesson": report["lesson"],
        "avoid_rule": report["avoid_rule"],
        "warning_signs": report.get("warning_signs", []),
    }])


def holding_period_review(trade: dict[str, Any], prices: pd.DataFrame, macro: pd.DataFrame,
                          *, mode: str = "backtest", use_memory: bool = True) -> str:
    """Ce qui s'est réellement passé entre l'entrée et la sortie (matière première de l'autopsie)."""
    entry, exit_ = pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade["exit_date"])
    window = prices.loc[entry:exit_]
    parts = []
    if len(window):
        low_day = window["low"].idxmin()
        parts.append(f"Plus bas atteint : {window['low'].min():.2f} USD le {low_day.date()} "
                     f"({window['low'].min() / trade['entry_price'] - 1:+.1%} vs prix d'entrée) ; "
                     f"plus haut : {window['high'].max():.2f} USD.")
    m = macro.ffill().loc[entry - pd.Timedelta(days=5):exit_]
    if len(m) > 1:
        if "spy" in m:
            parts.append(f"S&P 500 sur la période : {m['spy'].iloc[-1] / m['spy'].iloc[0] - 1:+.1%}.")
        if "vix" in m:
            parts.append(f"VIX : {m['vix'].iloc[0]:.1f} -> {m['vix'].iloc[-1]:.1f} (max {m['vix'].max():.1f}).")
    if use_memory:
        try:
            if mode == "backtest":
                fetch_news.ensure_historical_context(trade["ticker"], exit_)
            ts_from, _ = _day_bounds(entry)
            _, ts_to = _day_bounds(exit_)
            hits = semantic_search(
                NEWS_COLLECTION, f"{fetch_news.company_name(trade['ticker'])} {trade['ticker']} stock falls: bad news, "
                "earnings miss, downgrade, lawsuit, market selloff",
                tickers=[trade["ticker"], fetch_news.MACRO], ts_min=ts_from, ts_max=ts_to, limit=3)
            if hits:
                parts.append("Dépêches publiées pendant la détention :\n" + format_news(hits))
        except Exception as exc:
            log.warning("Dépêches de la période de détention indisponibles : %s", exc)
    return " ".join(parts) or "Aucune information disponible sur la période de détention."


# ---------------------------------------------------------------------------
# Dossier complet
# ---------------------------------------------------------------------------
def build_investment_dossier(ticker: str, as_of: pd.Timestamp, prices: pd.DataFrame, macro: pd.DataFrame,
                             *, mode: str = "backtest", scope: Optional[str] = None,
                             use_memory: bool = True) -> dict[str, Any]:
    """Assemble le dossier remis aux agents. ``prices`` et ``macro`` s'arrêtent à ``as_of``."""
    as_of = pd.Timestamp(as_of).normalize()
    if prices.index[-1] > as_of or (len(macro) and macro.index[-1] > as_of):
        raise ValueError(f"Fuite temporelle : des données postérieures au {as_of.date()} ont été transmises.")

    technical = technical_snapshot(prices)
    macro_text = macro_snapshot(macro)
    alt = alternative_snapshot(ticker, as_of)
    if alt:
        technical = f"{technical} {alt}"

    news_hits: list[dict] = []
    lessons: list[dict] = []
    if use_memory:
        ensure_collections()
        if mode == "backtest":
            fetch_news.ensure_historical_context(ticker, as_of)
        news_hits = get_contextual_news(ticker, as_of)
        lessons = get_experience_lessons(f"{ticker}. {technical} {macro_text}", as_of, scope)

    return {
        "ticker": ticker,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "technical": technical,
        "macro": macro_text,
        "fundamentals": fundamentals_snapshot(ticker, as_of, float(prices["close"].iloc[-1]), mode),
        "news": format_news(news_hits),
        "experience": format_lessons(lessons),
        "news_items": news_hits,
        "lesson_items": lessons,
    }
