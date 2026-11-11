"""Texte intégral des dépôts SEC (8-K, 10-Q, 10-K), daté à la minute de sa mise en ligne.

``fetch_news.py`` ne stocke que le titre des dépôts pour le RAG. Ici on télécharge les documents
eux-mêmes pour les faire lire par un LLM (``llm_agents/filing_extractor.py``) :

  - 8-K   : le formulaire et ses annexes EX-99 (communiqué de résultats, présentation) ;
  - 10-Q  : rapport de gestion (MD&A), litiges et facteurs de risque ;
  - 10-K  : mêmes sections du rapport annuel.

Chaque dépôt porte ``available`` : la première séance à la clôture de laquelle il était lisible.
Un communiqué publié à 16 h 30 n'est donc utilisable qu'à partir du lendemain.

Usage :
    python -m data_pipeline.fetch_filings --tickers AAPL MSFT --start 2022-01-01
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from typing import Iterable, Optional

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from data_pipeline.fetch_history import HISTORY_START, TICKERS, db_connect
from data_pipeline.fetch_news import _company_map, _sec_get

log = logging.getLogger("filings")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # documents XBRL en ligne

FORMS = ("8-K", "10-Q", "10-K")
MARKET_TZ = "America/New_York"
MAX_CHARS = int(os.getenv("FILING_MAX_CHARS", "80000"))   # plafond de texte conservé par dépôt
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY,
    ticker TEXT NOT NULL, form TEXT NOT NULL,
    filed TEXT NOT NULL,            -- date de dépôt
    accepted TEXT NOT NULL,         -- horodatage UTC de mise en ligne
    available TEXT NOT NULL,        -- première séance où le dépôt est connu à la clôture
    items TEXT, url TEXT,
    raw_chars INTEGER, text TEXT
);
CREATE INDEX IF NOT EXISTS filings_ticker ON filings (ticker, available);
"""

# Sections lues dans les rapports périodiques : (nom, motif du titre, plafond de caractères)
SECTIONS = {
    "10-Q": [
        ("MD&A", r"item\s*2\s*[.:\-–—]?\s*management.s\s+discussion", 55_000),
        ("LEGAL PROCEEDINGS", r"item\s*1\s*[.:\-–—]?\s*legal\s+proceedings", 10_000),
        ("RISK FACTORS", r"item\s*1a\s*[.:\-–—]?\s*risk\s+factors", 15_000),
    ],
    "10-K": [
        ("MD&A", r"item\s*7\s*[.:\-–—]?\s*management.s\s+discussion", 55_000),
        ("LEGAL PROCEEDINGS", r"item\s*3\s*[.:\-–—]?\s*legal\s+proceedings", 10_000),
        ("RISK FACTORS", r"item\s*1a\s*[.:\-–—]?\s*risk\s+factors", 15_000),
    ],
}
ITEM_HEADING = re.compile(r"\n\s*(?:part\s+[iv]+\s*[.:\-–—]?\s*)?item\s*\d+[a-c]?\s*[.:\-–—]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Disponibilité (anti-lookahead)
# ---------------------------------------------------------------------------
def available_date(accepted_utc: str) -> str:
    """Première date à la clôture de laquelle un dépôt accepté à ``accepted_utc`` est connu.

    Accepté jusqu'à 16 h (New York) : le jour même. Après la clôture : le lendemain. La date est
    ensuite recalée sur la prochaine séance réelle au moment de construire les features.
    """
    local = pd.Timestamp(accepted_utc).tz_convert(MARKET_TZ) if pd.Timestamp(accepted_utc).tzinfo \
        else pd.Timestamp(accepted_utc, tz="UTC").tz_convert(MARKET_TZ)
    day = local.normalize()
    if (local.hour, local.minute, local.second) > (16, 0, 0):
        day += pd.Timedelta(days=1)
    return day.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# HTML -> texte
# ---------------------------------------------------------------------------
def html_to_text(html: str) -> str:
    """Texte lisible d'un document EDGAR : tableaux aplatis ligne par ligne, balisage XBRL retiré."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", ")", "("}]
        row.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5"]):
        block.append("\n")
    text = soup.get_text().replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_cover(text: str) -> str:
    """Retire la page de garde d'un 8-K (adresse, cases à cocher) : le contenu commence au premier « Item x.xx »."""
    match = re.search(r"\n\s*item\s+\d\.\d{2}", "\n" + text, re.IGNORECASE)
    return text[match.start():].strip() if match else text


def select_sections(text: str, form: str, max_chars: int = MAX_CHARS) -> str:
    """Extrait d'un 10-Q / 10-K les sections utiles. Sans titre reconnu, garde le début du document.

    Un titre apparaît deux fois (sommaire puis corps) : on retient l'occurrence suivie du plus long
    texte avant le prochain « Item ».
    """
    padded = "\n" + text
    headings = [m.start() for m in ITEM_HEADING.finditer(padded)] + [len(padded)]
    parts = []
    for name, pattern, cap in SECTIONS.get(form, []):
        best = ""
        for match in re.finditer(r"\n\s*(?:part\s+[iv]+\s*[.:\-–—]?\s*)?" + pattern, padded, re.IGNORECASE):
            end = next(h for h in headings if h > match.start())
            body = padded[match.start():end].strip()
            if len(body) > len(best):
                best = body
        if len(best) > 400:  # en dessous, c'est une ligne de sommaire ou un renvoi
            parts.append(f"=== {name} ===\n{best[:cap]}")
    selected = "\n\n".join(parts)
    if len(selected) < 8_000 < len(text):
        # Mise en page non standard (ex. banques, qui renvoient à un index) : on lit le début du rapport,
        # où figurent le résumé des résultats et le rapport de gestion.
        selected = (selected + "\n\n=== REPORT (beginning) ===\n" + text).strip()
    return selected[:max_chars]


# ---------------------------------------------------------------------------
# EDGAR
# ---------------------------------------------------------------------------
def list_filings(ticker: str, start: str, end: Optional[str] = None, forms: Iterable[str] = FORMS) -> list[dict]:
    """Dépôts d'une société sur la période (API ``submissions``, environ dix ans d'historique)."""
    info = _company_map().get(ticker.upper())
    if not info:
        log.warning("%s : société inconnue de la SEC.", ticker)
        return []
    filings = _sec_get(f"https://data.sec.gov/submissions/CIK{info['cik']:010d}.json").json()["filings"]
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    # Les sociétés très actives (banques) débordent de la page « recent » : on lit aussi les pages d'archive utiles.
    pages = [filings["recent"]]
    for extra in filings.get("files", []):
        if extra["filingTo"] >= start and extra["filingFrom"] <= end:
            pages.append(_sec_get(f"https://data.sec.gov/submissions/{extra['name']}").json())
    wanted = set(forms)
    rows = []
    for page in pages:
        for i, form in enumerate(page["form"]):
            filed = page["filingDate"][i]
            if form not in wanted or not start <= filed <= end:
                continue
            accession = page["accessionNumber"][i]
            folder = f"{ARCHIVES}/{info['cik']}/{accession.replace('-', '')}"
            rows.append({
                "accession": accession, "ticker": ticker.upper(), "form": form, "filed": filed,
                "accepted": page["acceptanceDateTime"][i], "available": available_date(page["acceptanceDateTime"][i]),
                "items": page["items"][i], "folder": folder, "primary": page["primaryDocument"][i],
            })
    return sorted(rows, key=lambda r: r["accepted"])


def exhibit_urls(index_html: str) -> list[str]:
    """Adresses des annexes EX-99 (communiqués, présentations) listées sur la page d'index d'un dépôt."""
    urls = []
    for row in BeautifulSoup(index_html, "lxml").select("table.tableFile tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        link = row.find("a")
        if len(cells) >= 4 and cells[3].upper().startswith("EX-99") and link is not None:
            href = link["href"].replace("/ix?doc=", "")
            if href.lower().endswith((".htm", ".html", ".txt")):
                urls.append("https://www.sec.gov" + href)
    return urls


def download_text(filing: dict) -> tuple[str, int]:
    """(texte retenu, taille brute) d'un dépôt."""
    primary = html_to_text(_sec_get(f"{filing['folder']}/{filing['primary']}").text)
    if filing["form"] != "8-K":
        return select_sections(primary, filing["form"]), len(primary)
    parts = [f"=== FORM 8-K ===\n{strip_cover(primary)}"]
    try:
        index = _sec_get(f"{filing['folder']}/{filing['accession']}-index.htm").text
        for n, url in enumerate(exhibit_urls(index)[:3], 1):
            parts.append(f"=== EXHIBIT 99 ({n}) ===\n{html_to_text(_sec_get(url).text)}")
    except Exception as exc:  # l'annexe est un plus : le formulaire seul reste exploitable
        log.warning("%s : annexes illisibles (%s).", filing["accession"], exc)
    text = "\n\n".join(parts)
    return text[:MAX_CHARS], len(text)


def sync_filings(tickers: Iterable[str], start: str = HISTORY_START, end: Optional[str] = None,
                 forms: Iterable[str] = FORMS, limit: Optional[int] = None) -> int:
    """Télécharge les dépôts manquants. Retourne le nombre de documents ajoutés."""
    added = 0
    with db_connect() as con:
        con.executescript(SCHEMA)
        known = {row[0] for row in con.execute("SELECT accession FROM filings")}
        for ticker in tickers:
            todo = [f for f in list_filings(ticker, start, end, forms) if f["accession"] not in known]
            for filing in todo[:limit]:
                try:
                    text, raw_chars = download_text(filing)
                except Exception as exc:
                    log.warning("%s %s du %s : téléchargement impossible (%s).", ticker, filing["form"],
                                filing["filed"], exc)
                    continue
                con.execute("INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    filing["accession"], filing["ticker"], filing["form"], filing["filed"], filing["accepted"],
                    filing["available"], filing["items"], f"{filing['folder']}/{filing['primary']}", raw_chars, text))
                con.commit()
                added += 1
            log.info("%s : %d dépôt(s) ajouté(s).", ticker, min(len(todo), limit or len(todo)))
    return added


def load_filings(tickers: Optional[Iterable[str]] = None, start: Optional[str] = None, end: Optional[str] = None,
                 forms: Optional[Iterable[str]] = None, with_text: bool = True) -> pd.DataFrame:
    columns = "accession, ticker, form, filed, accepted, available, items, url, raw_chars" + (", text" if with_text else "")
    with db_connect() as con:
        con.executescript(SCHEMA)
        frame = pd.read_sql_query(f"SELECT {columns} FROM filings ORDER BY accepted", con)
    if tickers is not None:
        frame = frame[frame["ticker"].isin([t.upper() for t in tickers])]
    if forms is not None:
        frame = frame[frame["form"].isin(list(forms))]
    if start:
        frame = frame[frame["filed"] >= start]
    if end:
        frame = frame[frame["filed"] <= end]
    return frame.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge le texte intégral des dépôts SEC.")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--start", default=HISTORY_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--forms", nargs="*", default=list(FORMS))
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximal de dépôts par société.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = sync_filings([t.upper() for t in args.tickers], args.start, args.end, args.forms, args.limit)
    stored = load_filings(with_text=False)
    print(f"{count} dépôt(s) téléchargé(s). En base : {len(stored)} "
          f"({', '.join(f'{n} {f}' for f, n in stored['form'].value_counts().items())}).")
