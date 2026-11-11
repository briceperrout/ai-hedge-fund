"""Verrou Pydantic : le CRO ne peut qu'autoriser ou bloquer, dans un format strict."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_agents.schemas import AutopsyReport, VetoDecision

APPROVE = {
    "veto_activated": False, "final_action": "BUY", "systemic_risk_detected": False, "risk_level": "LOW",
    "confidence_score": 0.8, "key_risks": ["Valorisation élevée"],
    "justification": "RSI à 38 et aucune dépêche négative dans le dossier.",
}
VETO = {**APPROVE, "veto_activated": True, "final_action": "HOLD", "risk_level": "HIGH"}


def test_valid_approval_and_veto():
    assert VetoDecision.model_validate(APPROVE).final_action == "BUY"
    assert VetoDecision.model_validate(VETO).veto_activated is True


@pytest.mark.parametrize("override", [
    {"veto_activated": True, "final_action": "BUY"},            # veto mais achat
    {"veto_activated": False, "final_action": "HOLD"},          # pas de veto mais pas d'achat
    {"systemic_risk_detected": True},                           # risque systémique sans veto
    {"final_action": "SELL"},                                   # action hors périmètre
    {"risk_level": "EXTREME"},                                  # niveau inconnu
    {"confidence_score": 1.4},                                  # hors [0, 1]
    {"confidence_score": -0.1},
    {"justification": "ok"},                                    # justification creuse
    {"position_size": 0.5},                                     # le LLM tente de fixer la taille
    {"stop_loss": 95.0},                                        # ... ou le stop
])
def test_incoherent_or_out_of_scope_output_is_rejected(override):
    with pytest.raises(ValidationError):
        VetoDecision.model_validate({**APPROVE, **override})


@pytest.mark.parametrize("missing", list(APPROVE))
def test_every_field_is_required(missing):
    payload = {k: v for k, v in APPROVE.items() if k != missing}
    with pytest.raises(ValidationError):
        VetoDecision.model_validate(payload)


def test_key_risks_are_cleaned_and_capped():
    decision = VetoDecision.model_validate({**APPROVE, "key_risks": [" a ", "", "b", "c", "d", "e"]})
    assert decision.key_risks == ["a", "b", "c", "d"]


def test_fail_safe_is_a_non_systemic_veto():
    decision = VetoDecision.fail_safe("modèle éteint")
    assert decision.veto_activated and decision.final_action == "HOLD"
    assert decision.systemic_risk_detected is False  # une panne ne doit pas liquider le portefeuille
    assert "modèle éteint" in decision.justification


def test_autopsy_report_rejects_unknown_category_and_extra_fields():
    valid = {"error_category": "MACRO_REGIME", "root_cause": "Hausse brutale du VIX pendant le trade.",
             "warning_signs": ["VIX en backwardation"], "lesson": "Ne pas acheter quand le VIX s'inverse.",
             "avoid_rule": "Bloquer le trade si VIX/VIX3M > 1."}
    assert AutopsyReport.model_validate(valid).error_category == "MACRO_REGIME"
    with pytest.raises(ValidationError):
        AutopsyReport.model_validate({**valid, "error_category": "UNKNOWN"})
    with pytest.raises(ValidationError):
        AutopsyReport.model_validate({**valid, "new_order": "BUY TSLA"})
