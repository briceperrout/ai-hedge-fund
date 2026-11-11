"""Ingestion et vectorisation des flux d'information (RSS + SEC) dans Qdrant.

Sources :
  - Flux RSS : Yahoo Finance et Google News par ticker, CNBC / MarketWatch / Fed pour la macro.
  - SEC EDGAR : 8-K, 10-Q, 10-K, Form 4 (transactions d'initiés, « délit d'initié légal »)
    et chiffres des rapports trimestriels (API XBRL ``companyfacts``).

Chaque document porte un ``timestamp`` de publication : c'est lui qui permet au backtest
de n'interroger que le passé (aucune dépêche future ne peut atteindre le CRO).

Usage :
    python -m data_pipeline.fetch_news                 # ingestion quotidienne (univers du .env)
    python -m data_pipeline.fetch_news --tickers AAPL --no-sec
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import feedparser
import pandas as pd
import requests
from dotenv import load_dotenv

from data_pipeline.fetch_history import TICKERS, db_connect
from data_pipeline.qdrant_setup import NEWS_COLLECTION, ensure_collections, upsert_documents

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger("news")

MACRO = "MACRO"
RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (ai-hedge-fund-local)"}
SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "ai-hedge-fund-local research-bot")}

TICKER_FEEDS = {
    "yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
}
MACRO_FEEDS = {
    "cnbc_economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "cnbc_markets": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "federal_reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
}
GOOGLE_NEWS = "https://news.google.com/rss/search"
MACRO_QUERY = "stock market Federal Reserve inflation recession economy"

SEC_FORMS = {
    "8-K": "current report (material event)",
    "10-Q": "quarterly report",
    "10-K": "annual report",
}
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "RevenuesNetOfInterestExpense",
]
FUNDAMENTAL_TAGS = {
    "revenue": REVENUE_TAGS,
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def _clean(text: str, limit: int = 600) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"&[a-z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _epoch(entry) -> Optional[int]:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return int(calendar.timegm(parsed)) if parsed else None


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _already_ingested(source: str, ticker: str, window: str) -> bool:
    with db_connect() as con:
        row = con.execute(
            "SELECT 1 FROM ingestion_log WHERE source=? AND ticker=? AND window=?", (source, ticker, window)
        ).fetchone()
    return row is not None


def _mark_ingested(source: str, ticker: str, window: str) -> None:
    with db_connect() as con:
        con.execute("INSERT OR IGNORE INTO ingestion_log VALUES (?,?,?)", (source, ticker, window))
        con.commit()


def _week_window(as_of: pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """Semaine ISO contenant ``as_of`` : clé de cache + bornes (lundi, dimanche)."""
    monday = (as_of - pd.Timedelta(days=as_of.weekday())).normalize()
    iso = as_of.isocalendar()
    return f"{iso.year}-W{iso.week:02d}", monday, monday + pd.Timedelta(days=6)


# ---------------------------------------------------------------------------
# Flux RSS
# ---------------------------------------------------------------------------
def _fetch_feed(url: str, params: Optional[dict] = None):
    resp = requests.get(url, params=params, headers=RSS_HEADERS, timeout=20)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _entries_to_docs(feed, ticker: str, source: str, use_summary: bool = True) -> list[dict]:
    docs = []
    for entry in feed.entries:
        ts = _epoch(entry)
        title = _clean(entry.get("title", ""))
        if ts is None or not title:  # sans date fiable, impossible de garantir l'absence de lookahead
            continue
        summary = _clean(entry.get("summary", "")) if use_summary else ""
        text = title if not summary or summary.startswith(title[:40]) else f"{title}. {summary}"
        docs.append({
            "key": entry.get("link") or entry.get("id") or f"{source}:{ticker}:{title}",
            "text": text,
            "title": title,
            "ticker": ticker,
            "source": source,
            "kind": "news",
            "date": _iso(ts),
            "timestamp": ts,
            "link": entry.get("link", ""),
        })
    return docs


def ingest_rss(tickers: Iterable[str]) -> int:
    """Ingestion quotidienne des dépêches fraîches (par ticker + macro)."""
    docs: list[dict] = []
    for ticker in tickers:
        for source, url in TICKER_FEEDS.items():
            try:
                docs += _entries_to_docs(_fetch_feed(url.format(ticker=ticker)), ticker, source)
            except Exception as exc:
                log.warning("Flux %s indisponible pour %s : %s", source, ticker, exc)
        try:
            query = f"{company_name(ticker)} {ticker} stock when:7d"
            feed = _fetch_feed(GOOGLE_NEWS, {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
            docs += _entries_to_docs(feed, ticker, "google_news", use_summary=False)
        except Exception as exc:
            log.warning("Google News indisponible pour %s : %s", ticker, exc)
    for source, url in MACRO_FEEDS.items():
        try:
            docs += _entries_to_docs(_fetch_feed(url), MACRO, source)
        except Exception as exc:
            log.warning("Flux macro %s indisponible : %s", source, exc)
    return upsert_documents(NEWS_COLLECTION, docs)


def backfill_news(ticker: str, as_of: pd.Timestamp) -> int:
    """Reconstitue les dépêches d'une semaine passée (backtest) via la recherche datée Google News."""
    window, monday, sunday = _week_window(pd.Timestamp(as_of))
    if _already_ingested("google_news", ticker, window):
        return 0
    start = (monday - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (sunday + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    topic = MACRO_QUERY if ticker == MACRO else f"{company_name(ticker)} {ticker} stock"
    try:
        feed = _fetch_feed(GOOGLE_NEWS, {"q": f"{topic} after:{start} before:{end}",
                                         "hl": "en-US", "gl": "US", "ceid": "US:en"})
    except Exception as exc:
        log.warning("Backfill Google News impossible (%s, %s) : %s", ticker, window, exc)
        return 0
    count = upsert_documents(NEWS_COLLECTION, _entries_to_docs(feed, ticker, "google_news", use_summary=False))
    _mark_ingested("google_news", ticker, window)
    return count


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------
_last_sec_call = 0.0


def _sec_get(url: str, params: Optional[dict] = None) -> requests.Response:
    """GET respectueux de la limite SEC (10 requêtes/s)."""
    global _last_sec_call
    wait = 0.15 - (time.monotonic() - _last_sec_call)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, params=params, headers=SEC_HEADERS, timeout=30)
    _last_sec_call = time.monotonic()
    resp.raise_for_status()
    return resp


def _company_map() -> dict[str, dict]:
    """Table ticker -> {cik, name}, mise en cache 30 jours."""
    cache = ROOT / "data" / "sec_company_tickers.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 30 * 86400:
        return json.loads(cache.read_text())
    try:
        raw = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
    except Exception as exc:
        log.warning("Table des sociétés SEC indisponible : %s", exc)
        return json.loads(cache.read_text()) if cache.exists() else {}
    mapping = {row["ticker"].upper(): {"cik": int(row["cik_str"]), "name": row["title"]} for row in raw.values()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapping))
    return mapping


def company_name(ticker: str) -> str:
    return _company_map().get(ticker.upper(), {}).get("name", ticker)


def _parse_form4(xml_text: str, company: str, ticker: str) -> Optional[str]:
    """Résume un Form 4 : qui, achat ou vente sur le marché, combien."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    owner = (root.findtext(".//rptOwnerName") or "An insider").title()
    role = root.findtext(".//officerTitle") or ("Director" if root.findtext(".//isDirector") in ("1", "true") else "Insider")
    bought = sold = other = 0.0
    bought_value = sold_value = 0.0
    last_date = root.findtext(".//periodOfReport") or ""
    for tx in root.iter("nonDerivativeTransaction"):
        code = tx.findtext(".//transactionCode") or ""
        try:
            shares = float(tx.findtext(".//transactionShares/value") or 0)
            price = float(tx.findtext(".//transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        if code == "P":
            bought += shares
            bought_value += shares * price
        elif code == "S":
            sold += shares
            sold_value += shares * price
        else:
            other += shares
    parts = []
    if bought:
        parts.append(f"bought {bought:,.0f} shares on the open market (~${bought_value:,.0f})")
    if sold:
        parts.append(f"sold {sold:,.0f} shares on the open market (~${sold_value:,.0f})")
    if not parts:
        if not other:
            return None
        parts.append(f"reported {other:,.0f} shares in non-market transactions (grant, option exercise or tax withholding)")
    return (f"Insider transaction (SEC Form 4) at {company} ({ticker}): {owner} ({role}) "
            f"{' and '.join(parts)} on {last_date}.")


def _edgar_feed(cik: int, form: str, before: Optional[pd.Timestamp], count: int, owner: str = "include"):
    params = {"action": "getcompany", "CIK": f"{cik:010d}", "type": form, "owner": owner,
              "count": count, "output": "atom"}
    if before is not None:
        params["dateb"] = before.strftime("%Y%m%d")
    return feedparser.parse(_sec_get("https://www.sec.gov/cgi-bin/browse-edgar", params).content)


def ingest_sec_filings(ticker: str, as_of: Optional[pd.Timestamp] = None, lookback_days: int = 45,
                       max_form4: int = 6) -> int:
    """Ingestion des dépôts SEC récents (ou de la semaine de ``as_of`` pour le backtest)."""
    info = _company_map().get(ticker.upper())
    if not info:
        return 0
    before = None
    if as_of is not None:
        window, _, sunday = _week_window(pd.Timestamp(as_of))
        if _already_ingested("sec", ticker, window):
            return 0
        before = sunday
    floor = int(((before or pd.Timestamp.today()) - pd.Timedelta(days=lookback_days)).timestamp())
    cik, name = info["cik"], info["name"]
    docs: list[dict] = []
    try:
        for form, label in SEC_FORMS.items():
            for entry in _edgar_feed(cik, form, before, 8).entries:
                ts = _epoch(entry)
                if ts is None or ts < floor or not entry.get("title", "").strip().startswith(form):
                    continue
                items = "; ".join(re.findall(r"Item [\d.]+: [^<]+", entry.get("summary", "")))
                text = f"SEC filing {form} ({label}) by {name} ({ticker}) filed on {_iso(ts)}."
                if items:
                    text += f" {items}."
                docs.append({"key": entry.link, "text": text, "title": f"{ticker} {form}", "ticker": ticker,
                             "source": "sec_edgar", "kind": "sec_filing", "date": _iso(ts),
                             "timestamp": ts, "link": entry.link})
        # owner="only" : uniquement les déclarations d'initiés (sinon "4" ramène aussi les 424B2, 425...)
        for entry in _edgar_feed(cik, "4", before, max_form4, owner="only").entries:
            ts = _epoch(entry)
            if ts is None or ts < floor or not re.match(r"4(/A)?\s", entry.get("title", "").strip()):
                continue
            base = entry.link.rsplit("/", 1)[0]
            files = _sec_get(f"{base}/index.json").json()["directory"]["item"]
            xml_name = next((f["name"] for f in files if f["name"].endswith(".xml")), None)
            if not xml_name:
                continue
            text = _parse_form4(_sec_get(f"{base}/{xml_name}").text, name, ticker)
            if text:
                docs.append({"key": entry.link, "text": text, "title": f"{ticker} Form 4", "ticker": ticker,
                             "source": "sec_edgar", "kind": "insider", "date": _iso(ts),
                             "timestamp": ts, "link": entry.link})
    except Exception as exc:
        log.warning("SEC EDGAR indisponible pour %s : %s", ticker, exc)
        return 0
    count = upsert_documents(NEWS_COLLECTION, docs)
    if as_of is not None:
        _mark_ingested("sec", ticker, window)
    return count


def sync_fundamentals(ticker: str, force: bool = False) -> int:
    """Chiffres des rapports trimestriels/annuels (SEC XBRL), datés de leur publication.

    Écrit dans SQL (``fundamentals``) pour des fondamentaux « point-in-time » et dans
    Qdrant (kind=quarterly_report) pour le RAG.
    """
    window = pd.Timestamp.today().strftime("%Y-%m")
    info = _company_map().get(ticker.upper())
    if not info or (not force and _already_ingested("sec_facts", ticker, window)):
        return 0
    try:
        facts = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{info['cik']:010d}.json").json()
        gaap = facts["facts"]["us-gaap"]
    except Exception as exc:
        log.warning("Rapports trimestriels SEC indisponibles pour %s : %s", ticker, exc)
        return 0

    best: dict[tuple[str, str, str], dict] = {}
    for metric, tags in FUNDAMENTAL_TAGS.items():
        for tag in tags:
            for unit_rows in gaap.get(tag, {}).get("units", {}).values():
                for row in unit_rows:
                    if "start" not in row or row.get("form") not in ("10-Q", "10-K"):
                        continue
                    days = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
                    if not (80 <= days <= 100 or 350 <= days <= 380):
                        continue
                    key = (metric, row["start"], row["end"])
                    # première publication du chiffre = date à laquelle le marché l'a connu
                    if key not in best or row["filed"] < best[key]["filed"]:
                        best[key] = row
    rows = [(ticker, metric, start, end, float(r["val"]), r["form"], r["filed"])
            for (metric, start, end), r in best.items()]
    with db_connect() as con:
        con.executemany("INSERT OR REPLACE INTO fundamentals VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()

    # Un document RAG par rapport publié
    reports: dict[tuple[str, str, str], dict] = {}
    for (metric, start, end), r in best.items():
        reports.setdefault((r["filed"], start, end), {"form": r["form"]})[metric] = float(r["val"])
    docs = []
    for (filed, start, end), vals in reports.items():
        if "revenue" not in vals:
            continue
        kind = "annual" if (pd.Timestamp(end) - pd.Timestamp(start)).days > 300 else "quarterly"
        text = (f"{kind.capitalize()} report ({vals['form']}) of {info['name']} ({ticker}) filed on {filed} "
                f"for the period {start} to {end}: revenue ${vals['revenue'] / 1e9:.2f}B")
        if "net_income" in vals:
            text += f", net income ${vals['net_income'] / 1e9:.2f}B (net margin {vals['net_income'] / vals['revenue']:.1%})"
        if "eps_diluted" in vals:
            text += f", diluted EPS ${vals['eps_diluted']:.2f}"
        ts = int(pd.Timestamp(filed, tz="UTC").timestamp()) + 21 * 3600  # après la clôture US
        docs.append({"key": f"sec-facts:{ticker}:{start}:{end}", "text": text + ".", "title": f"{ticker} {vals['form']}",
                     "ticker": ticker, "source": "sec_xbrl", "kind": "quarterly_report", "date": filed,
                     "timestamp": ts, "link": ""})
    upsert_documents(NEWS_COLLECTION, docs)
    _mark_ingested("sec_facts", ticker, window)
    return len(rows)


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------
def ensure_historical_context(ticker: str, as_of: pd.Timestamp, with_sec: bool = True) -> None:
    """Backtest : garantit que la mémoire contient l'information disponible autour de ``as_of``."""
    backfill_news(ticker, as_of)
    backfill_news(MACRO, as_of)
    if with_sec:
        ingest_sec_filings(ticker, as_of=as_of)
        sync_fundamentals(ticker)


def ingest_daily(tickers: Iterable[str], with_sec: bool = True) -> dict[str, int]:
    """Live : ingestion quotidienne complète."""
    tickers = list(tickers)
    ensure_collections()
    report = {"rss": ingest_rss(tickers), "sec_filings": 0, "fundamentals": 0}
    if with_sec:
        for ticker in tickers:
            report["sec_filings"] += ingest_sec_filings(ticker)
            report["fundamentals"] += sync_fundamentals(ticker)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Ingestion quotidienne des dépêches et dépôts SEC dans Qdrant.")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--no-sec", action="store_true", help="Ne pas interroger SEC EDGAR.")
    args = parser.parse_args()
    result = ingest_daily(args.tickers, with_sec=not args.no_sec)
    print(f"Dépêches RSS vectorisées : {result['rss']} | dépôts SEC : {result['sec_filings']} | "
          f"lignes de fondamentaux : {result['fundamentals']}")
