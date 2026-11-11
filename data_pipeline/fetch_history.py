"""Alimentation des prix (YFinance) et entrepôt SQL local (SQLite).

Tout l'historique chiffré du fonds vit dans ``data/market.db`` :
  - ``prices``        : OHLCV ajustés (actions + indicateurs macro : VIX, SPY, taux...)
  - ``dark_pool``     : volumes hors-marché FINRA (proxy Dark Pools, Phase 3)
  - ``options_flow``  : instantanés Put/Call des chaînes d'options (Phase 3)
  - ``fundamentals``  : rapports trimestriels SEC datés (point-in-time)
  - ``trades``        : journal de trading (backtest et live)

Usage :
    python -m data_pipeline.fetch_history                       # univers du .env
    python -m data_pipeline.fetch_history --tickers AAPL MSFT --dark-pool --options
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger("history")

DB_PATH = ROOT / "data" / "market.db"
HISTORY_START = os.getenv("HISTORY_START", "2016-01-01")
TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "AAPL").split(",") if t.strip()]

# Indicateurs macro-économiques et de marché (symbole Yahoo -> nom de colonne)
MACRO_TICKERS = {
    "^VIX": "vix",      # indice de peur
    "^VIX3M": "vix3m",  # volatilité implicite 3 mois (structure par terme)
    "^SKEW": "skew",    # prix des puts de queue (flux d'options institutionnels)
    "^VVIX": "vvix",    # volatilité de la volatilité
    "^TNX": "tnx",      # taux 10 ans US
    "SPY": "spy",       # S&P 500
}

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day}.txt"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (ai-hedge-fund-local)"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS splits (
    ticker TEXT NOT NULL, date TEXT NOT NULL, ratio REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS dark_pool (
    ticker TEXT NOT NULL, date TEXT NOT NULL,
    short_volume REAL, total_volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS options_flow (
    ticker TEXT NOT NULL, date TEXT NOT NULL,
    call_volume REAL, put_volume REAL, call_oi REAL, put_oi REAL,
    pc_volume_ratio REAL, pc_oi_ratio REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT NOT NULL, metric TEXT NOT NULL,
    period_start TEXT NOT NULL, period_end TEXT NOT NULL,
    value REAL, form TEXT, filed TEXT NOT NULL,
    PRIMARY KEY (ticker, metric, period_start, period_end)
);
CREATE TABLE IF NOT EXISTS ingestion_log (
    source TEXT NOT NULL, ticker TEXT NOT NULL, window TEXT NOT NULL,
    PRIMARY KEY (source, ticker, window)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL, mode TEXT NOT NULL, ticker TEXT NOT NULL,
    signal_date TEXT, entry_date TEXT, entry_price REAL, shares REAL,
    stop_loss REAL, take_profit REAL, probability REAL,
    kelly_fraction REAL, allocation_pct REAL,
    exit_date TEXT, exit_price REAL, exit_reason TEXT,
    pnl REAL, pnl_pct REAL,
    context_json TEXT, autopsy_done INTEGER NOT NULL DEFAULT 0
);
"""


def db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# ---------------------------------------------------------------------------
# Prix (YFinance)
# ---------------------------------------------------------------------------
def _bounds(con: sqlite3.Connection, ticker: str) -> tuple[Optional[str], Optional[str]]:
    row = con.execute("SELECT MIN(date), MAX(date) FROM prices WHERE ticker = ?", (ticker,)).fetchone()
    return row[0], row[1]


def sync_prices(tickers: Iterable[str], start: str = HISTORY_START, end: Optional[str] = None,
                force: bool = False) -> dict[str, int]:
    """Télécharge l'historique ajusté et le range en base.

    ``end=None`` (mode live) force un rafraîchissement jusqu'à la dernière séance.
    Avec ``end`` fixé (backtest), un ticker déjà couvert en base n'est pas re-téléchargé.
    La série complète est toujours remplacée d'un bloc pour garder des prix ajustés cohérents.
    """
    report: dict[str, int] = {}
    with db_connect() as con:
        for ticker in tickers:
            lo, hi = _bounds(con, ticker)
            if (not force and end is not None and lo and hi
                    and pd.Timestamp(lo) <= pd.Timestamp(start) + pd.Timedelta(days=10)
                    and pd.Timestamp(hi) >= pd.Timestamp(end) - pd.Timedelta(days=5)):
                report[ticker] = 0
                continue
            dl_start = min(start, lo) if lo else start
            try:
                df = yf.Ticker(ticker).history(start=dl_start, auto_adjust=True)
            except Exception as exc:
                log.error("Téléchargement impossible pour %s : %s", ticker, exc)
                continue
            if df is None or df.empty:
                log.warning("Aucune donnée Yahoo pour %s", ticker)
                continue
            df = df.rename(columns=str.lower)
            split_rows = []
            if "stock splits" in df:
                events = df.loc[df["stock splits"] > 0, "stock splits"]
                split_rows = [(ticker, pd.Timestamp(d).tz_localize(None).strftime("%Y-%m-%d"), float(r))
                              for d, r in events.items()]
            df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            rows = [
                (ticker, d.strftime("%Y-%m-%d"), float(o), float(h), float(l), float(c), float(v))
                for d, o, h, l, c, v in zip(idx, df["open"], df["high"], df["low"], df["close"], df["volume"])
            ]
            con.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
            con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", rows)
            con.executemany("INSERT OR REPLACE INTO splits VALUES (?,?,?)", split_rows)
            con.commit()
            report[ticker] = len(rows)
            log.info("%-6s : %d séances (%s -> %s)", ticker, len(rows), rows[0][1], rows[-1][1])
    return report


def sync_all(tickers: Optional[Iterable[str]] = None, start: str = HISTORY_START,
             end: Optional[str] = None, force: bool = False) -> dict[str, int]:
    """Actions de l'univers + indicateurs macro."""
    universe = list(tickers or TICKERS)
    return sync_prices(universe + list(MACRO_TICKERS), start=start, end=end, force=force)


def load_prices(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    query = "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ?"
    params: list = [ticker]
    if start:
        query += " AND date >= ?"
        params.append(str(start)[:10])
    if end:
        query += " AND date <= ?"
        params.append(str(end)[:10])
    with db_connect() as con:
        df = pd.read_sql_query(query + " ORDER BY date", con, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def split_factor(ticker: str, after: str, until: Optional[str] = None) -> float:
    """Produit des divisions d'actions survenues dans l'intervalle ]after, until].

    Les prix en base sont ajustés de tous les splits ; les BPA des rapports SEC sont, eux,
    publiés dans l'unité de leur époque. Ce facteur permet de les rendre comparables.
    """
    query, params = "SELECT ratio FROM splits WHERE ticker = ? AND date > ?", [ticker, str(after)[:10]]
    if until is not None:
        query += " AND date <= ?"
        params.append(str(until)[:10])
    with db_connect() as con:
        rows = con.execute(query, params).fetchall()
    factor = 1.0
    for (ratio,) in rows:
        factor *= ratio
    return factor


def load_macro(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Tableau macro quotidien (clôtures) : vix, vix3m, skew, vvix, tnx, spy."""
    frames = {}
    for symbol, name in MACRO_TICKERS.items():
        series = load_prices(symbol, start, end)["close"]
        if not series.empty:
            frames[name] = series
    return pd.DataFrame(frames).sort_index()


# ---------------------------------------------------------------------------
# Phase 3 — Dark Pools (volumes hors-marché FINRA Reg SHO)
# ---------------------------------------------------------------------------
def _fetch_finra_day(day: pd.Timestamp, wanted: set[str]) -> tuple[str, dict[str, tuple[float, float]]]:
    date = day.strftime("%Y-%m-%d")
    found: dict[str, tuple[float, float]] = {}
    try:
        resp = requests.get(FINRA_URL.format(day=day.strftime("%Y%m%d")), headers=HTTP_HEADERS, timeout=30)
        if resp.status_code == 200:
            for line in resp.text.splitlines()[1:]:
                parts = line.split("|")
                if len(parts) >= 5 and parts[1] in wanted:
                    found[parts[1]] = (float(parts[2]), float(parts[4]))
    except requests.RequestException as exc:
        log.debug("FINRA %s : %s", date, exc)
    return date, found


def sync_dark_pool(tickers: Iterable[str], start: str, end: Optional[str] = None, workers: int = 8) -> int:
    """Télécharge les fichiers quotidiens FINRA (ventes à découvert exécutées hors-marché).

    Le ratio ``short_volume / total_volume`` hors-marché sert de proxy d'activité Dark Pool.
    Un fichier par séance (~350 Ko) : comptez plusieurs minutes pour plusieurs années.
    """
    wanted = {t.upper() for t in tickers}
    last = min(pd.Timestamp(end) if end else pd.Timestamp.today(), pd.Timestamp.today() - pd.Timedelta(days=2))
    days = pd.bdate_range(max(pd.Timestamp(start), pd.Timestamp("2018-08-01")), last.normalize())
    with db_connect() as con:
        done = pd.read_sql_query(
            f"SELECT date, COUNT(*) AS n FROM dark_pool WHERE ticker IN ({','.join('?' * len(wanted))}) GROUP BY date",
            con, params=sorted(wanted),
        )
        complete = set(done.loc[done["n"] >= len(wanted), "date"])
        todo = [d for d in days if d.strftime("%Y-%m-%d") not in complete]
        if not todo:
            return 0
        log.info("FINRA : %d fichier(s) quotidien(s) à télécharger...", len(todo))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda d: _fetch_finra_day(d, wanted), todo))
        rows = []
        for date, found in results:
            for ticker in wanted:  # ligne NULL = jour férié / ticker absent -> ne sera pas re-téléchargé
                short_vol, total_vol = found.get(ticker, (None, None))
                rows.append((ticker, date, short_vol, total_vol))
        con.executemany("INSERT OR REPLACE INTO dark_pool VALUES (?,?,?,?)", rows)
        con.commit()
    return len(todo)


def load_dark_pool(ticker: str) -> pd.Series:
    """Ratio quotidien de ventes à découvert hors-marché (0..1)."""
    with db_connect() as con:
        df = pd.read_sql_query(
            "SELECT date, short_volume, total_volume FROM dark_pool "
            "WHERE ticker = ? AND total_volume > 0 ORDER BY date", con, params=[ticker],
        )
    if df.empty:
        return pd.Series(dtype=float, name="dark_short_ratio")
    series = (df["short_volume"] / df["total_volume"]).rename("dark_short_ratio")
    series.index = pd.to_datetime(df["date"])
    return series


# ---------------------------------------------------------------------------
# Phase 3 — Flux d'options (instantané Put/Call, historisé jour après jour)
# ---------------------------------------------------------------------------
def snapshot_options_flow(ticker: str, max_expiries: int = 3) -> Optional[dict]:
    """Agrège volumes et positions ouvertes des échéances proches et les stocke en base.

    Yahoo ne fournit pas d'historique des chaînes d'options : c'est en lançant ce relevé
    chaque jour (``run_live.py`` le fait) que l'historique se construit.
    """
    try:
        tk = yf.Ticker(ticker)
        totals = {"call_volume": 0.0, "put_volume": 0.0, "call_oi": 0.0, "put_oi": 0.0}
        for expiry in list(tk.options)[:max_expiries]:
            chain = tk.option_chain(expiry)
            totals["call_volume"] += float(chain.calls["volume"].fillna(0).sum())
            totals["put_volume"] += float(chain.puts["volume"].fillna(0).sum())
            totals["call_oi"] += float(chain.calls["openInterest"].fillna(0).sum())
            totals["put_oi"] += float(chain.puts["openInterest"].fillna(0).sum())
    except Exception as exc:
        log.warning("Chaîne d'options indisponible pour %s : %s", ticker, exc)
        return None
    if totals["call_volume"] <= 0 or totals["call_oi"] <= 0:
        return None
    snap = {
        "ticker": ticker,
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        **totals,
        "pc_volume_ratio": totals["put_volume"] / totals["call_volume"],
        "pc_oi_ratio": totals["put_oi"] / totals["call_oi"],
    }
    with db_connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO options_flow VALUES "
            "(:ticker,:date,:call_volume,:put_volume,:call_oi,:put_oi,:pc_volume_ratio,:pc_oi_ratio)", snap,
        )
        con.commit()
    return snap


def load_options_flow(ticker: str) -> pd.DataFrame:
    with db_connect() as con:
        df = pd.read_sql_query(
            "SELECT date, pc_volume_ratio, pc_oi_ratio FROM options_flow WHERE ticker = ? ORDER BY date",
            con, params=[ticker],
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Alimente la base SQL locale en prix et données alternatives.")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--start", default=HISTORY_START)
    parser.add_argument("--end", default=None, help="Laisser vide pour aller jusqu'à la dernière séance.")
    parser.add_argument("--force", action="store_true", help="Re-télécharge même si la base est à jour.")
    parser.add_argument("--dark-pool", action="store_true", help="Télécharge aussi les volumes hors-marché FINRA.")
    parser.add_argument("--options", action="store_true", help="Relève l'instantané Put/Call du jour.")
    args = parser.parse_args()

    summary = sync_all(args.tickers, start=args.start, end=args.end, force=args.force)
    print(f"Prix : {sum(1 for n in summary.values() if n)} série(s) mise(s) à jour, "
          f"{sum(1 for n in summary.values() if not n)} déjà en base.")
    if args.dark_pool:
        print(f"Dark pool : {sync_dark_pool(args.tickers, args.start, args.end)} fichier(s) FINRA intégré(s).")
    if args.options:
        for tkr in args.tickers:
            snap = snapshot_options_flow(tkr)
            if snap:
                print(f"Options {tkr} : Put/Call volume = {snap['pc_volume_ratio']:.2f}, OI = {snap['pc_oi_ratio']:.2f}")
