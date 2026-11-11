"""Contrats de données : état du graphe (TypedDict) et sorties LLM validées (Pydantic).

Le LLM ne produit que des recommandations typées. Toute sortie qui ne respecte pas
ces schémas est rejetée ; le code Python (et lui seul) calcule et exécute l'ordre.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentState(TypedDict, total=False):
    # --- Signal du Cerveau Gauche ---
    ticker: str
    as_of: str
    trigger_event: str
    statistical_action: str
    win_probability: float
    # --- Dossier d'investissement ---
    technical_snapshot: str
    fundamentals: str
    macro_environment: str
    news_summary: str
    experience_lessons: str
    # --- Débat ---
    bull_thesis: str
    bear_thesis: str
    # --- Arbitrage ---
    decision: dict
    errors: Annotated[list[str], operator.add]


class AutopsyState(TypedDict, total=False):
    trade: dict
    report: Optional[dict]
    errors: Annotated[list[str], operator.add]


class VetoDecision(BaseModel):
    """Verdict du Chief Risk Officer : il autorise ou bloque le signal, rien d'autre."""

    model_config = ConfigDict(extra="forbid")

    veto_activated: bool = Field(
        description="true pour BLOQUER le trade (risque macro, actualité ou fondamental majeur). false pour l'autoriser.")
    final_action: Literal["BUY", "HOLD"] = Field(
        description="'HOLD' si veto_activated est true, sinon 'BUY' (le signal statistique est repris tel quel).")
    systemic_risk_detected: bool = Field(
        description="true uniquement en cas de risque systémique (krach, crise bancaire, choc géopolitique ou de liquidité) "
                    "justifiant aussi de couper les positions ouvertes.")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Niveau de risque global du dossier.")
    confidence_score: float = Field(description="Confiance dans ce verdict, entre 0.0 et 1.0.")
    key_risks: list[str] = Field(description="1 à 4 risques concrets identifiés dans le dossier (phrases courtes).")
    justification: str = Field(description="Justification du verdict en 2 phrases maximum, citant un fait du dossier.")

    @field_validator("confidence_score")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence_score doit être compris entre 0.0 et 1.0")
        return value

    @field_validator("justification")
    @classmethod
    def _justification_present(cls, value: str) -> str:
        if len(value.strip()) < 15:
            raise ValueError("justification trop courte : citer un fait précis du dossier")
        return value.strip()

    @field_validator("key_risks")
    @classmethod
    def _risks_bounded(cls, value: list[str]) -> list[str]:
        return [risk.strip() for risk in value if risk.strip()][:4]

    @model_validator(mode="after")
    def _coherence(self) -> "VetoDecision":
        if self.veto_activated and self.final_action != "HOLD":
            raise ValueError("incohérence : veto_activated=true impose final_action='HOLD'")
        if not self.veto_activated and self.final_action != "BUY":
            raise ValueError("incohérence : sans veto, final_action doit valoir 'BUY' (tu ne peux qu'autoriser ou bloquer)")
        if self.systemic_risk_detected and not self.veto_activated:
            raise ValueError("incohérence : un risque systémique impose veto_activated=true")
        return self

    @classmethod
    def fail_safe(cls, reason: str) -> "VetoDecision":
        """Verdict de sécurité : sans décision valide du CRO, aucun trade ne passe."""
        return cls(veto_activated=True, final_action="HOLD", systemic_risk_detected=False, risk_level="HIGH",
                   confidence_score=0.0, key_risks=["Décision du CRO indisponible"],
                   justification=f"Veto de sécurité automatique : {reason}")


class AutopsyReport(BaseModel):
    """Analyse d'un trade perdant, stockée dans le journal de trading (RAG d'expérience)."""

    model_config = ConfigDict(extra="forbid")

    error_category: Literal[
        "MACRO_REGIME", "NEWS_BLIND_SPOT", "EARNINGS_EVENT", "TECHNICAL_FALSE_SIGNAL", "OVERCONFIDENCE", "BAD_LUCK",
    ] = Field(description="Famille d'erreur. BAD_LUCK uniquement si rien dans le dossier ne permettait d'anticiper la perte.")
    root_cause: str = Field(description="Cause principale de la perte, en 1 phrase factuelle.")
    warning_signs: list[str] = Field(description="1 à 3 signaux présents dans le dossier d'entrée qui auraient dû alerter.")
    lesson: str = Field(description="Leçon structurelle réutilisable sur d'autres valeurs, en 1 phrase.")
    avoid_rule: str = Field(description="Règle opérationnelle : 'Bloquer le trade si ...'.")

    @field_validator("root_cause", "lesson", "avoid_rule")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError("champ trop court")
        return value.strip()

    @field_validator("warning_signs")
    @classmethod
    def _signs_bounded(cls, value: list[str]) -> list[str]:
        return [sign.strip() for sign in value if sign.strip()][:3]


class FilingSignals(BaseModel):
    """Variables chiffrées extraites d'un dépôt SEC (8-K, 10-Q, 10-K) par le LLM.

    Le LLM est ici une source d'information : il transforme un texte long en quelques nombres
    que XGBoost peut exploiter. Les deux premiers champs chiffrés sont vérifiables contre les
    données XBRL officielles, et chaque citation contre le texte source.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: Literal[
        "EARNINGS", "GUIDANCE_UPDATE", "M_AND_A", "MANAGEMENT_CHANGE", "FINANCING", "LEGAL_REGULATORY",
        "RESTRUCTURING", "SHAREHOLDER_RETURN", "OTHER",
    ] = Field(description="Nature principale du document. EARNINGS pour une publication de résultats ou un rapport périodique.")
    revenue_musd: Optional[float] = Field(
        description="Chiffre d'affaires du TRIMESTRE publié, en MILLIONS de dollars (ex. 117154 pour 117,154 milliards). "
                    "null si le document ne le donne pas.")
    eps_diluted: Optional[float] = Field(
        description="Bénéfice par action dilué du TRIMESTRE publié, en dollars (ex. 1.88). null s'il est absent.")
    revenue_growth_yoy_pct: Optional[float] = Field(
        description="Variation du chiffre d'affaires trimestriel sur un an, en pourcentage (ex. -5.5). null si absente.")
    guidance_direction: int = Field(
        description="Prévisions de la direction : 1 relevées ou optimistes, -1 abaissées ou prudentes, 0 inchangées ou absentes.")
    demand_tone: int = Field(
        description="Ton sur la demande et l'activité, de -2 (forte dégradation) à 2 (forte amélioration), 0 si neutre ou sans objet.")
    margin_trend: int = Field(description="Marges : 1 en hausse, -1 en baisse, 0 stables ou non mentionnées.")
    liquidity_risk: int = Field(
        description="Tension sur la trésorerie ou la dette, de 0 (aucune) à 3 (risque de continuité d'exploitation).")
    legal_risk: int = Field(
        description="Litiges et risque réglementaire NOUVEAUX ou aggravés, de 0 (rien de notable) à 3 (menace majeure).")
    shareholder_return: int = Field(
        description="Rachats d'actions et dividendes : 1 annonce ou hausse, -1 suspension ou baisse, 0 inchangé ou non mentionné.")
    evidence: list[str] = Field(
        description="1 à 3 citations EXACTES du document (copiées mot pour mot, en anglais, 25 mots maximum chacune) "
                    "qui justifient les notes ci-dessus.")
    summary: str = Field(description="L'essentiel du document en une phrase.")

    @field_validator("guidance_direction", "margin_trend", "shareholder_return")
    @classmethod
    def _direction(cls, value: int) -> int:
        if value not in (-1, 0, 1):
            raise ValueError("doit valoir -1, 0 ou 1")
        return value

    @field_validator("demand_tone")
    @classmethod
    def _tone(cls, value: int) -> int:
        if not -2 <= value <= 2:
            raise ValueError("doit être compris entre -2 et 2")
        return value

    @field_validator("liquidity_risk", "legal_risk")
    @classmethod
    def _risk(cls, value: int) -> int:
        if not 0 <= value <= 3:
            raise ValueError("doit être compris entre 0 et 3")
        return value

    @field_validator("revenue_musd")
    @classmethod
    def _revenue_positive(cls, value: Optional[float]) -> Optional[float]:
        if value == 0:  # « 0 » est la façon dont les petits modèles écrivent « absent »
            return None
        if value is not None and value <= 0:
            raise ValueError("un chiffre d'affaires est strictement positif (null s'il est absent)")
        if value is not None and value > 2_000_000:  # aucune société ne réalise 2 000 milliards par trimestre
            raise ValueError("valeur invraisemblable : le chiffre d'affaires doit être exprimé en MILLIONS de dollars "
                             "(117154 pour 117,154 milliards), pas en dollars")
        return value

    @field_validator("eps_diluted")
    @classmethod
    def _eps_plausible(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and abs(value) > 1_000:
            raise ValueError("valeur invraisemblable : le bénéfice par action s'exprime en dollars par action (ex. 1.88)")
        return value

    @field_validator("evidence")
    @classmethod
    def _evidence_present(cls, value: list[str]) -> list[str]:
        quotes = [quote.strip().strip('"“”') for quote in value if quote.strip()][:3]
        if not quotes:
            raise ValueError("au moins une citation exacte du document est exigée")
        return quotes
