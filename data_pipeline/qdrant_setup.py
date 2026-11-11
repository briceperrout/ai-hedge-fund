"""Initialisation des collections vectorielles Qdrant (la mémoire du fonds).

Deux collections :
  - ``financial_news`` : RAG d'information (dépêches RSS, formulaires SEC, rapports trimestriels).
  - ``trade_journal``  : RAG d'expérience (autopsies des trades perdants).

Usage :
    python -m data_pipeline.qdrant_setup            # crée les collections manquantes
    python -m data_pipeline.qdrant_setup --recreate # repart d'une base vide
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

# macOS : PyTorch (embeddings) et XGBoost embarquent chacun leur propre runtime OpenMP ; les faire
# tourner en multi-thread dans le même processus provoque un plantage (segfault). Un seul thread
# OpenMP par bibliothèque supprime le conflit. Doit être fixé avant le chargement de ces bibliothèques.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger("qdrant")

NEWS_COLLECTION = "financial_news"
JOURNAL_COLLECTION = "trade_journal"
COLLECTIONS = (NEWS_COLLECTION, JOURNAL_COLLECTION)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_STORE = ROOT / "data" / "qdrant_local"


# ---------------------------------------------------------------------------
# Embeddings (Sentence-Transformers, 100 % local)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_encoder():
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    try:  # modèle déjà en cache : aucun appel réseau (fonctionnement 100 % local)
        return SentenceTransformer(EMBEDDING_MODEL, device=device, local_files_only=True)
    except Exception:  # premier lancement : téléchargement unique du modèle d'embedding
        return SentenceTransformer(EMBEDDING_MODEL, device=device)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Client Qdrant : serveur local (ports du .env) ou, à défaut, moteur embarqué sur disque."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    try:
        client = QdrantClient(host=host, port=port, timeout=5)
        client.get_collections()
        log.info("Qdrant : serveur %s:%s", host, port)
        return client
    except Exception as exc:  # serveur éteint
        if os.getenv("QDRANT_FALLBACK_LOCAL", "true").lower() != "true":
            raise RuntimeError(
                f"Qdrant injoignable sur {host}:{port} ({exc}). Lancez-le : "
                "docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant"
            ) from exc
        LOCAL_STORE.mkdir(parents=True, exist_ok=True)
        log.warning(
            "Qdrant injoignable sur %s:%s -> moteur Qdrant embarqué (%s). "
            "Un seul processus à la fois peut l'utiliser.", host, port, LOCAL_STORE,
        )
        client = QdrantClient(path=str(LOCAL_STORE))
        atexit.register(client.close)  # libère proprement le verrou du dossier
        return client


def ensure_collections(recreate: bool = False) -> None:
    """Crée (ou recrée) les collections et leurs index de filtrage."""
    client = get_client()
    dim = len(embed(["dimension probe"])[0])
    for name in COLLECTIONS:
        if recreate and client.collection_exists(name):
            client.delete_collection(name)
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            log.info("Collection créée : %s (dim=%d)", name, dim)
        for field, schema in (
            ("ticker", models.PayloadSchemaType.KEYWORD),
            ("kind", models.PayloadSchemaType.KEYWORD),
            ("timestamp", models.PayloadSchemaType.INTEGER),
        ):
            try:
                client.create_payload_index(name, field_name=field, field_schema=schema)
            except Exception:  # index déjà présent, ou moteur embarqué (index inutiles)
                pass


# ---------------------------------------------------------------------------
# Écriture / lecture
# ---------------------------------------------------------------------------
def stable_id(key: str) -> str:
    """Identifiant déterministe : ré-ingérer la même dépêche ne crée pas de doublon."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def upsert_documents(collection: str, docs: Iterable[dict[str, Any]], batch_size: int = 64) -> int:
    """Vectorise et insère des documents ``{"key": str, "text": str, **payload}``."""
    docs = list(docs)
    client = get_client()
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        vectors = embed([d["text"] for d in chunk])
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=stable_id(d["key"]),
                    vector=vec,
                    payload={k: v for k, v in d.items() if k != "key"},
                )
                for d, vec in zip(chunk, vectors)
            ],
        )
    return len(docs)


def semantic_search(
    collection: str,
    query: str,
    *,
    tickers: Optional[list[str]] = None,
    ts_min: Optional[int] = None,
    ts_max: Optional[int] = None,
    kinds: Optional[list[str]] = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Recherche sémantique filtrée. ``ts_max`` est le rempart anti-lookahead du backtest."""
    must: list[models.FieldCondition] = []
    if tickers:
        must.append(models.FieldCondition(key="ticker", match=models.MatchAny(any=tickers)))
    if kinds:
        must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=kinds)))
    if ts_min is not None or ts_max is not None:
        must.append(models.FieldCondition(key="timestamp", range=models.Range(gte=ts_min, lte=ts_max)))
    hits = get_client().query_points(
        collection_name=collection,
        query=embed([query])[0],
        query_filter=models.Filter(must=must) if must else None,
        limit=limit,
        with_payload=True,
    ).points
    return [{**(h.payload or {}), "score": float(h.score)} for h in hits]


def count_documents(collection: str, *, tickers: Optional[list[str]] = None,
                    ts_min: Optional[int] = None, ts_max: Optional[int] = None) -> int:
    must: list[models.FieldCondition] = []
    if tickers:
        must.append(models.FieldCondition(key="ticker", match=models.MatchAny(any=tickers)))
    if ts_min is not None or ts_max is not None:
        must.append(models.FieldCondition(key="timestamp", range=models.Range(gte=ts_min, lte=ts_max)))
    return get_client().count(
        collection, count_filter=models.Filter(must=must) if must else None, exact=True
    ).count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Initialise les collections Qdrant du fonds.")
    parser.add_argument("--recreate", action="store_true", help="Supprime puis recrée les collections.")
    args = parser.parse_args()
    ensure_collections(recreate=args.recreate)
    for name in COLLECTIONS:
        print(f"{name:<16} : {count_documents(name)} document(s)")
