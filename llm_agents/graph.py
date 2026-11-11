"""Le Cerveau Droit : topologie du réseau multi-agents (LangGraph, asynchrone).

    START ─┬─> Bull (Llama 3.1 8B / Ollama) ─┬─> CRO (Qwen 2.5 32B / LM Studio) ──> END
           └─> Bear (Llama 3.1 8B / Ollama) ─┘        └─ sortie verrouillée : VetoDecision

Le comité ne s'active que lorsqu'XGBoost a déjà validé un signal. Sans verdict valide
du CRO (service éteint, JSON invalide), le trade est bloqué : jamais l'inverse.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TypeVar

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from llm_agents.prompts import AUTOPSY_PROMPT, BEAR_PROMPT, BULL_PROMPT, CRO_PROMPT, RETRY_MESSAGE
from llm_agents.schemas import AgentState, AutopsyReport, AutopsyState, VetoDecision

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

log = logging.getLogger("agents")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
BULL_BEAR_MODEL = os.getenv("BULL_BEAR_MODEL", "llama3.1:8b")
CRO_MODEL = os.getenv("CRO_MODEL", "qwen2.5-32b-instruct")
STRUCTURED_METHOD = os.getenv("CRO_STRUCTURED_METHOD", "json_schema")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Le LLM n'a pas produit de JSON conforme au contrat Pydantic malgré les relances."""


# ---------------------------------------------------------------------------
# 1. Connexion aux modèles locaux
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def analyst_llm() -> ChatOllama:
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=BULL_BEAR_MODEL, temperature=0.7, num_predict=350,
                      client_kwargs={"timeout": LLM_TIMEOUT})


@lru_cache(maxsize=1)
def cro_llm() -> ChatOpenAI:
    return ChatOpenAI(base_url=LM_STUDIO_BASE_URL, api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                      model=CRO_MODEL, temperature=0.1, timeout=LLM_TIMEOUT, max_retries=1)


def check_llm_services() -> list[str]:
    """Liste les problèmes de connexion aux modèles locaux (liste vide = tout est prêt)."""
    problems = []
    try:
        tags = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5).json()
        names = {m["name"] for m in tags.get("models", [])}
        if BULL_BEAR_MODEL not in names and f"{BULL_BEAR_MODEL}:latest" not in names:
            problems.append(f"Ollama ne possède pas le modèle '{BULL_BEAR_MODEL}' (lancer : ollama pull {BULL_BEAR_MODEL}).")
    except Exception as exc:
        problems.append(f"Ollama injoignable sur {OLLAMA_BASE_URL} ({type(exc).__name__}).")
    try:
        headers = {"Authorization": f"Bearer {os.getenv('LM_STUDIO_API_KEY', 'lm-studio')}"}
        models = requests.get(f"{LM_STUDIO_BASE_URL.rstrip('/')}/models", headers=headers, timeout=5).json()
        ids = {m["id"] for m in models.get("data", [])}
        if ids and CRO_MODEL not in ids and f"{CRO_MODEL}:latest" not in ids:
            problems.append(f"LM Studio ne sert pas le modèle '{CRO_MODEL}' (modèles chargés : {', '.join(sorted(ids))}).")
    except Exception as exc:
        problems.append(f"LM Studio injoignable sur {LM_STUDIO_BASE_URL} ({type(exc).__name__}).")
    return problems


# ---------------------------------------------------------------------------
# 2. Verrouillage Pydantic de la sortie du CRO
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    return match.group(0) if match else (text or "")


async def structured_call(prompt: ChatPromptTemplate, variables: dict[str, Any], schema: type[T],
                          usage: Optional[dict[str, int]] = None) -> T:
    """Appelle le modèle d'arbitrage jusqu'à obtenir un objet conforme au schéma.

    Chaque rejet du validateur est renvoyé au modèle pour correction. Après
    ``MAX_RETRIES`` échecs, ``StructuredOutputError`` est levée. ``usage``, s'il est fourni, cumule
    les jetons consommés (clés ``input_tokens``, ``output_tokens``, ``calls``).
    """
    runnable = cro_llm().with_structured_output(schema, method=STRUCTURED_METHOD, include_raw=True)
    messages = prompt.format_messages(**variables)
    error = "aucune réponse"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await runnable.ainvoke(messages)
        except ValidationError as exc:
            # En mode json_schema, le SDK OpenAI valide lui-même la réponse et lève l'erreur avant que
            # LangChain ne la capture : on la traite comme un rejet ordinaire, donc avec relance.
            result = {"parsed": None, "raw": None, "validation_error": exc}
        if usage is not None:
            meta = getattr(result.get("raw"), "usage_metadata", None) or {}
            usage["input_tokens"] = usage.get("input_tokens", 0) + int(meta.get("input_tokens", 0))
            usage["output_tokens"] = usage.get("output_tokens", 0) + int(meta.get("output_tokens", 0))
            usage["calls"] = usage.get("calls", 0) + 1
        if result.get("parsed") is not None:
            return result["parsed"]
        raw = result.get("raw")
        raw_text = raw.content if raw is not None and isinstance(raw.content, str) else ""
        try:  # seconde chance : JSON entouré de texte ou de balises
            if result.get("validation_error") is not None:
                raise result["validation_error"]
            return schema.model_validate_json(_extract_json(raw_text))
        except ValidationError as exc:
            error = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'objet'} : {e['msg']}" for e in exc.errors())
        log.warning("Sortie %s rejetée (tentative %d/%d) : %s", schema.__name__, attempt, MAX_RETRIES, error)
        messages = messages + [HumanMessage(RETRY_MESSAGE.format(error=error[:600]))]
    raise StructuredOutputError(error)


# ---------------------------------------------------------------------------
# 3. Nœuds du comité d'investissement
# ---------------------------------------------------------------------------
def _dossier_vars(state: AgentState) -> dict[str, Any]:
    return {
        "ticker": state["ticker"],
        "as_of": state["as_of"],
        "trigger": state["trigger_event"],
        "technical": state["technical_snapshot"],
        "fundamental": state["fundamentals"],
        "macro": state["macro_environment"],
        "news": state["news_summary"],
        "win_probability": state["win_probability"],
    }


async def bull_node(state: AgentState) -> dict[str, Any]:
    response = await (BULL_PROMPT | analyst_llm()).ainvoke(_dossier_vars(state))
    return {"bull_thesis": str(response.content).strip()}


async def bear_node(state: AgentState) -> dict[str, Any]:
    response = await (BEAR_PROMPT | analyst_llm()).ainvoke(_dossier_vars(state))
    return {"bear_thesis": str(response.content).strip()}


async def cro_node(state: AgentState) -> dict[str, Any]:
    variables = {
        **_dossier_vars(state),
        "stat_action": state["statistical_action"],
        "experience": state["experience_lessons"],
        "bull": state["bull_thesis"],
        "bear": state["bear_thesis"],
    }
    try:
        decision = await structured_call(CRO_PROMPT, variables, VetoDecision)
        return {"decision": decision.model_dump()}
    except StructuredOutputError as exc:
        fallback = VetoDecision.fail_safe(f"JSON du CRO invalide après {MAX_RETRIES} tentatives.")
        return {"decision": fallback.model_dump(), "errors": [f"Verrou Pydantic : {exc}"]}


def build_committee_graph():
    builder = StateGraph(AgentState)
    builder.add_node("bull", bull_node)
    builder.add_node("bear", bear_node)
    builder.add_node("cro", cro_node)
    builder.add_edge(START, "bull")          # Bull et Bear rédigent en parallèle
    builder.add_edge(START, "bear")
    builder.add_edge(["bull", "bear"], "cro")  # le CRO attend les deux thèses
    builder.add_edge("cro", END)
    return builder.compile()


committee_app = build_committee_graph()


# ---------------------------------------------------------------------------
# 4. Agent d'Autopsie (apprentissage continu)
# ---------------------------------------------------------------------------
async def autopsy_node(state: AutopsyState) -> dict[str, Any]:
    trade = state["trade"]
    context = trade.get("context", {})
    variables = {
        "ticker": trade["ticker"],
        "entry_date": str(trade["entry_date"])[:10],
        "exit_date": str(trade["exit_date"])[:10],
        "entry_price": float(trade["entry_price"]),
        "exit_price": float(trade["exit_price"]),
        "exit_reason": trade["exit_reason"],
        "pnl_pct": float(trade["pnl_pct"]),
        "probability": float(trade["probability"]),
        "technical": context.get("technical", "n/d"),
        "fundamental": context.get("fundamentals", "n/d"),
        "macro": context.get("macro", "n/d"),
        "news": context.get("news", "n/d"),
        "bull": context.get("bull_thesis", "n/d"),
        "bear": context.get("bear_thesis", "n/d"),
        "cro_justification": context.get("cro_justification", "n/d"),
        "post_mortem": trade.get("post_mortem", "n/d"),
    }
    try:
        report = await structured_call(AUTOPSY_PROMPT, variables, AutopsyReport)
        return {"report": report.model_dump()}
    except StructuredOutputError as exc:
        return {"report": None, "errors": [f"Autopsie illisible : {exc}"]}


def build_autopsy_graph():
    builder = StateGraph(AutopsyState)
    builder.add_node("autopsy", autopsy_node)
    builder.add_edge(START, "autopsy")
    builder.add_edge("autopsy", END)
    return builder.compile()


autopsy_app = build_autopsy_graph()


# ---------------------------------------------------------------------------
# 5. Points d'entrée pour le code d'exécution
# ---------------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None


def _run_async(coroutine: Any) -> Any:
    """Exécute un graphe asynchrone depuis le code d'exécution synchrone.

    Une seule boucle d'événements pour toute la durée du processus : les clients HTTP des
    modèles y restent attachés (``asyncio.run`` fermerait la boucle après chaque signal).
    """
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coroutine)


def run_committee(signal: dict[str, Any], dossier: dict[str, Any]) -> dict[str, Any]:
    """Soumet un signal XGBoost au comité et retourne un verdict toujours exploitable.

    Retour : {"decision": VetoDecision, "bull_thesis", "bear_thesis", "error"}.
    ``error`` est renseigné si le comité n'a pas pu délibérer (le verdict est alors un veto de sécurité).
    """
    state: AgentState = {
        "ticker": dossier["ticker"],
        "as_of": dossier["as_of"],
        "trigger_event": (f"XGBoost estime à {signal['win_probability']:.0%} la probabilité que l'objectif de gain "
                          f"soit atteint avant le stop-loss."),
        "statistical_action": signal["statistical_action"],
        "win_probability": float(signal["win_probability"]),
        "technical_snapshot": dossier["technical"],
        "fundamentals": dossier["fundamentals"],
        "macro_environment": dossier["macro"],
        "news_summary": dossier["news"],
        "experience_lessons": dossier["experience"],
        "errors": [],
    }
    try:
        final = _run_async(committee_app.ainvoke(state))
        decision = VetoDecision.model_validate(final["decision"])
        error = "; ".join(final.get("errors", [])) or None
    except Exception as exc:  # modèle éteint, délai dépassé... -> on bloque
        log.error("Comité indisponible (%s) : %s", type(exc).__name__, exc)
        final = {}
        decision = VetoDecision.fail_safe("comité d'investissement indisponible.")
        error = f"{type(exc).__name__}: {exc}"

    # Muraille de Chine : le LLM ne peut pas créer un ordre que XGBoost n'a pas émis.
    if decision.final_action == "BUY" and signal["statistical_action"] != "BUY":
        decision = VetoDecision.fail_safe("le CRO a tenté d'acheter sans signal statistique.")
    return {
        "decision": decision,
        "bull_thesis": final.get("bull_thesis", ""),
        "bear_thesis": final.get("bear_thesis", ""),
        "error": error,
    }


def run_autopsy(trade: dict[str, Any]) -> Optional[AutopsyReport]:
    """Analyse un trade perdant. Retourne None si le modèle est indisponible ou illisible."""
    try:
        final = _run_async(autopsy_app.ainvoke({"trade": trade, "errors": []}))
    except Exception as exc:
        log.error("Autopsie impossible (%s) : %s", type(exc).__name__, exc)
        return None
    return AutopsyReport.model_validate(final["report"]) if final.get("report") else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    issues = check_llm_services()
    print(f"Bull/Bear : {BULL_BEAR_MODEL} @ {OLLAMA_BASE_URL}\nCRO       : {CRO_MODEL} @ {LM_STUDIO_BASE_URL}")
    print("\n".join(f"  PROBLÈME : {p}" for p in issues) if issues else "  Tous les modèles répondent.")
    if not issues:
        demo = run_committee(
            {"statistical_action": "BUY", "win_probability": 0.74},
            {"ticker": "DEMO", "as_of": "2023-03-10",
             "technical": "Prix 150.00 USD. RSI(14) : 38. Tendance de fond haussière (+6% vs SMA200).",
             "fundamentals": "Chiffre d'affaires trimestriel +8% sur un an, marge nette 24%.",
             "macro": "VIX : 24.8, régime tendu, +5.9 pt sur 5 séances. Structure par terme VIX/VIX3M : 1.04 (backwardation).",
             "news": "- [2023-03-10] (MACRO) Silicon Valley Bank collapses in biggest US bank failure since 2008.",
             "experience": "Aucune leçon comparable dans le journal de trading."},
        )
        print(json.dumps(demo["decision"].model_dump(), indent=2, ensure_ascii=False))
