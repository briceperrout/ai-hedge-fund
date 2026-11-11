"""Lecture intégrale des dépôts SEC par les LLM locaux : du texte long aux variables chiffrées.

Deux temps (map-reduce), pour qu'aucun document ne dépasse la fenêtre de contexte :

    1. notes    : le petit modèle (Ollama) lit chaque tranche du document et relève les faits, citations à l'appui ;
    2. synthèse : le modèle d'arbitrage (LM Studio) convertit ces notes en un objet ``FilingSignals`` validé par Pydantic.

Un document court est envoyé directement à l'étape 2. Chaque résultat est mis en cache en base avec son coût
(appels, jetons, secondes) et un contrôle d'ancrage : la part des citations réellement présentes dans le texte.

Usage :
    python -m llm_agents.filing_extractor --tickers AAPL --start 2023-01-01 --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable, Optional

import pandas as pd
from langchain_ollama import ChatOllama

from data_pipeline.fetch_filings import load_filings
from data_pipeline.fetch_history import TICKERS, db_connect
from data_pipeline.fetch_news import company_name
from llm_agents.graph import (
    BULL_BEAR_MODEL, CRO_MODEL, LLM_TIMEOUT, OLLAMA_BASE_URL, StructuredOutputError, _run_async, structured_call,
)
from llm_agents.prompts import FILING_EXTRACT_PROMPT, FILING_NOTES_PROMPT
from llm_agents.schemas import FilingSignals

log = logging.getLogger("filings.llm")

PROMPT_VERSION = "v1"                                        # à incrémenter si les prompts ou le schéma changent
NOTES_MODEL = os.getenv("FILING_NOTES_MODEL", BULL_BEAR_MODEL)
CHUNK_CHARS = int(os.getenv("FILING_CHUNK_CHARS", "12000"))  # ~3 000 jetons par tranche
NUM_CTX = int(os.getenv("FILING_NUM_CTX", "8192"))
MODEL_ID = f"{NOTES_MODEL}+{CRO_MODEL}"

SCHEMA = """
CREATE TABLE IF NOT EXISTS filing_signals (
    accession TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
    signals_json TEXT,              -- NULL si l'extraction a échoué
    grounding REAL,                 -- part des citations retrouvées mot pour mot dans le document
    chunks INTEGER, calls INTEGER, input_tokens INTEGER, output_tokens INTEGER, seconds REAL,
    error TEXT, created_at TEXT NOT NULL,
    PRIMARY KEY (accession, model, prompt_version)
);
"""


@lru_cache(maxsize=1)
def notes_llm() -> ChatOllama:
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=NOTES_MODEL, temperature=0.0, num_ctx=NUM_CTX,
                      num_predict=700, client_kwargs={"timeout": LLM_TIMEOUT})


# ---------------------------------------------------------------------------
# Outils purs (testés sans modèle)
# ---------------------------------------------------------------------------
def split_chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Découpe sur les fins de paragraphe, sans jamais perdre de texte."""
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        while len(paragraph) > size:  # paragraphe géant (tableau aplati) : coupe franche
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size])
            paragraph = paragraph[size:]
        if len(current) + len(paragraph) + 1 > size and current:
            chunks.append(current)
            current = ""
        current = f"{current}\n{paragraph}" if current else paragraph
    if current.strip():
        chunks.append(current)
    return chunks


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9$%.']+", " ", text).strip()


def grounding_rate(quotes: list[str], document: str) -> float:
    """Part des citations présentes mot pour mot (à la ponctuation et aux espaces près) dans le document."""
    if not quotes:
        return 0.0
    haystack = _normalize(document)
    found = sum(1 for quote in quotes if len(_normalize(quote)) >= 12 and _normalize(quote).rstrip(".") in haystack)
    return found / len(quotes)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_signals(filing: dict[str, Any]) -> dict[str, Any]:
    """Lit un dépôt et retourne l'enregistrement à mettre en cache (jamais d'exception)."""
    started = time.monotonic()
    usage: dict[str, int] = {}
    meta = {"company": company_name(filing["ticker"]), "ticker": filing["ticker"], "form": filing["form"],
            "filed": filing["filed"]}
    chunks = split_chunks(filing["text"])
    record: dict[str, Any] = {"accession": filing["accession"], "model": MODEL_ID, "prompt_version": PROMPT_VERSION,
                              "signals_json": None, "grounding": None, "chunks": len(chunks), "error": None}
    try:
        if len(chunks) <= 1:
            content, label = filing["text"], "DOCUMENT"
        else:
            notes = []
            for part, chunk in enumerate(chunks, 1):
                reply = (FILING_NOTES_PROMPT | notes_llm()).invoke({**meta, "part": part, "parts": len(chunks),
                                                                   "text": chunk})
                tokens = getattr(reply, "usage_metadata", None) or {}
                usage["input_tokens"] = usage.get("input_tokens", 0) + int(tokens.get("input_tokens", 0))
                usage["output_tokens"] = usage.get("output_tokens", 0) + int(tokens.get("output_tokens", 0))
                usage["calls"] = usage.get("calls", 0) + 1
                body = str(reply.content).strip()
                if body and not body.upper().startswith("RIEN"):
                    notes.append(f"[Extrait {part}/{len(chunks)}]\n{body}")
            content, label = "\n\n".join(notes) or "Aucun fait utile relevé.", "NOTES DE LECTURE DU DOCUMENT"
        signals = _run_async(structured_call(FILING_EXTRACT_PROMPT, {**meta, "source_label": label, "content": content},
                                             FilingSignals, usage=usage))
        record["signals_json"] = signals.model_dump_json()
        record["grounding"] = grounding_rate(signals.evidence, filing["text"])
    except StructuredOutputError as exc:
        record["error"] = f"JSON invalide : {exc}"[:500]
    except Exception as exc:  # modèle éteint, délai dépassé...
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    record.update(calls=usage.get("calls", 0), input_tokens=usage.get("input_tokens", 0),
                  output_tokens=usage.get("output_tokens", 0), seconds=round(time.monotonic() - started, 1),
                  created_at=datetime.now().isoformat(timespec="seconds"))
    return record


def extract_all(tickers: Optional[Iterable[str]] = None, start: Optional[str] = None, end: Optional[str] = None,
                forms: Optional[Iterable[str]] = None, limit: Optional[int] = None, retry_errors: bool = False,
                earnings_only: bool = False, cheapest_first: bool = False, max_hours: Optional[float] = None) -> int:
    """Lit tous les dépôts téléchargés qui ne l'ont pas encore été par ce couple de modèles.

    ``earnings_only`` se limite aux communiqués de résultats (8-K rubrique 2.02), ``cheapest_first`` lit les
    documents les plus courts d'abord, ``max_hours`` arrête proprement après cette durée. La lecture s'interrompt
    aussi après trois pannes consécutives du modèle, pour ne pas enregistrer une nuit d'échecs.
    """
    filings = load_filings(tickers, start, end, forms)
    if earnings_only:
        filings = filings[filings["form"].eq("8-K") & filings["items"].fillna("").str.contains("2.02", regex=False)]
    if cheapest_first:
        filings = filings.loc[filings["text"].str.len().sort_values(kind="stable").index]
    deadline = time.monotonic() + max_hours * 3600 if max_hours else None
    outages, read = 0, 0
    with db_connect() as con:
        con.executescript(SCHEMA)
        done = {row[0] for row in con.execute(
            "SELECT accession FROM filing_signals WHERE model = ? AND prompt_version = ?"
            + ("" if not retry_errors else " AND error IS NULL"), (MODEL_ID, PROMPT_VERSION))}
    todo = filings[~filings["accession"].isin(done)]
    todo = todo.head(limit) if limit else todo
    for n, (_, filing) in enumerate(todo.iterrows(), 1):
        if deadline is not None and time.monotonic() > deadline:
            log.info("Durée maximale atteinte : arrêt après %d dépôt(s), le reste sera lu au prochain lancement.", read)
            break
        record = extract_signals(filing.to_dict())
        outage = record["error"] is not None and not record["error"].startswith("JSON invalide")
        outages = outages + 1 if outage else 0
        if outages >= 3:  # modèle injoignable : on n'enregistre pas ces échecs, ils seraient pris pour des lectures
            log.error("Trois pannes consécutives du modèle (%s) : arrêt.", record["error"])
            break
        if outage:
            continue
        read += 1
        with db_connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO filing_signals VALUES (:accession, :model, :prompt_version, :signals_json, "
                ":grounding, :chunks, :calls, :input_tokens, :output_tokens, :seconds, :error, :created_at)", record)
            con.commit()
        status = record["error"] or f"ancrage {record['grounding']:.0%}"
        log.info("[%d/%d] %s %s du %s : %d tranche(s), %.0f s, %s", n, len(todo), filing["ticker"], filing["form"],
                 filing["filed"], record["chunks"], record["seconds"], status)
    return read


def load_signals(model: Optional[str] = None, include_errors: bool = False) -> pd.DataFrame:
    """Dépôts et variables extraites, une ligne par dépôt lu (colonnes de ``FilingSignals`` dépliées)."""
    with db_connect() as con:
        con.executescript(SCHEMA)
        try:
            frame = pd.read_sql_query(
                "SELECT f.accession, f.ticker, f.form, f.filed, f.accepted, f.available, f.items, f.raw_chars, "
                "LENGTH(f.text) AS chars, s.model, s.signals_json, s.grounding, s.chunks, s.calls, s.input_tokens, "
                "s.output_tokens, s.seconds, s.error FROM filing_signals s JOIN filings f USING (accession) "
                "WHERE s.prompt_version = ? ORDER BY f.accepted", con, params=(PROMPT_VERSION,))
        except Exception:  # table ``filings`` absente : rien n'a encore été téléchargé
            return pd.DataFrame()
    if model:
        frame = frame[frame["model"] == model]
    if not include_errors:
        frame = frame[frame["signals_json"].notna()]
    parsed = pd.DataFrame([json.loads(js) if isinstance(js, str) else {} for js in frame["signals_json"]], index=frame.index)
    return pd.concat([frame.drop(columns=["signals_json"]), parsed], axis=1).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrait des variables chiffrées des dépôts SEC téléchargés.")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--forms", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximal de dépôts à lire.")
    parser.add_argument("--retry-errors", action="store_true", help="Relire les dépôts dont l'extraction a échoué.")
    parser.add_argument("--earnings-only", action="store_true", help="Seulement les communiqués de résultats (8-K 2.02).")
    parser.add_argument("--cheapest-first", action="store_true", help="Lire les documents les plus courts d'abord.")
    parser.add_argument("--max-hours", type=float, default=None, help="Arrêter proprement après cette durée.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Modèles : notes = {NOTES_MODEL} (Ollama), synthèse = {CRO_MODEL} (LM Studio)")
    count = extract_all([t.upper() for t in args.tickers], args.start, args.end, args.forms, args.limit,
                        args.retry_errors, args.earnings_only, args.cheapest_first, args.max_hours)
    print(f"{count} dépôt(s) lu(s).")
