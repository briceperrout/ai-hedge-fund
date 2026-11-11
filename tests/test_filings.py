"""Dépôts SEC lus par le LLM : texte, datation, schéma, ancrage, features point-in-time, évaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import run_filings_study as study
from data_pipeline import fetch_filings as ff
from llm_agents import filing_extractor as fx
from llm_agents.schemas import FilingSignals
from quant_engine.filing_features import FILING_FEATURES, features_from_signals

SIGNALS = {
    "event_type": "EARNINGS", "revenue_musd": 117154.0, "eps_diluted": 1.88, "revenue_growth_yoy_pct": -5.5,
    "guidance_direction": 0, "demand_tone": 1, "margin_trend": -1, "liquidity_risk": 0, "legal_risk": 1,
    "shareholder_return": 1, "evidence": ["Services set new all-time revenue record"],
    "summary": "Résultats du premier trimestre fiscal 2023.",
}


# --- Datation ---------------------------------------------------------------
@pytest.mark.parametrize("accepted, expected", [
    ("2023-02-02T21:30:33.000Z", "2023-02-03"),     # 16 h 30 à New York : après la clôture -> lendemain
    ("2023-01-13T11:45:00.000Z", "2023-01-13"),     # 6 h 45 : avant l'ouverture -> jour même
    ("2023-02-02T21:00:00.000Z", "2023-02-02"),     # 16 h 00 pile
    ("2023-02-03T23:01:00.000Z", "2023-02-04"),     # 18 h 01
    ("2023-07-14T19:59:00.000Z", "2023-07-14"),     # heure d'été : 15 h 59
    ("2023-07-14T20:01:00.000Z", "2023-07-15"),     # heure d'été : 16 h 01
])
def test_filing_is_known_only_after_it_was_published(accepted, expected):
    assert ff.available_date(accepted) == expected


# --- Texte ------------------------------------------------------------------
def test_html_to_text_flattens_tables_and_drops_hidden_markup():
    html = """<html><head><title>x</title><style>p{}</style></head><body>
      <ix:header><span>hidden xbrl</span></ix:header><div style="display:none">secret</div>
      <p>Revenue&nbsp;was <b>strong</b>.</p>
      <table><tr><td>Net sales</td><td>$</td><td>117,154</td></tr><tr><td>EPS</td><td>1.88</td></tr></table>
      <script>alert(1)</script></body></html>"""
    text = ff.html_to_text(html)
    assert "Revenue was strong." in text
    assert "Net sales | 117,154" in text and "EPS | 1.88" in text
    assert "hidden" not in text and "secret" not in text and "alert" not in text


def test_cover_page_of_an_8k_is_removed():
    text = "UNITED STATES\nFORM 8-K\n☐ Written communications\nItem 2.02 Results of Operations.\nOn February 2..."
    assert ff.strip_cover(text).startswith("Item 2.02")
    assert ff.strip_cover("no item here") == "no item here"


def test_sections_are_taken_from_the_body_not_the_table_of_contents():
    body = "Net sales decreased 5 percent. " * 60
    risks = "There have been no material changes. " * 30
    text = ("TABLE OF CONTENTS\nItem 2. Management's Discussion and Analysis 12\nItem 1A. Risk Factors 20\n\n"
            "Item 1. Financial Statements\nBalance sheet...\n"
            f"Item 2. Management’s Discussion and Analysis of Financial Condition\n{body}\n"
            "Item 3. Quantitative Disclosures\nMarket risk...\n"
            "PART II\nItem 1. Legal Proceedings\nshort\n"
            f"Item 1A. Risk Factors\n{risks}\nItem 6. Exhibits\n")
    selected = ff.select_sections(text, "10-Q")
    assert selected.startswith("=== MD&A ===")
    assert selected.count("Net sales decreased") == 60
    assert "=== RISK FACTORS ===" in selected and "no material changes" in selected
    assert "Balance sheet" not in selected and "Market risk" not in selected
    assert "LEGAL PROCEEDINGS" not in selected               # section trop courte pour être un vrai contenu


def test_non_standard_report_falls_back_to_the_beginning_of_the_document():
    text = "JPMorgan Chase & Co. consolidated financial highlights. " * 400
    selected = ff.select_sections(text, "10-Q", max_chars=5_000)
    assert "REPORT (beginning)" in selected and len(selected) == 5_000


def test_exhibits_are_found_on_the_index_page():
    index = """<table class="tableFile">
      <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
      <tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/2/a.htm">a.htm</a></td><td>8-K</td><td>1</td></tr>
      <tr><td>2</td><td>Press release</td><td><a href="/Archives/edgar/data/1/2/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>1</td></tr>
      <tr><td>3</td><td>Chart</td><td><a href="/Archives/edgar/data/1/2/g1.jpg">g1.jpg</a></td><td>EX-99.2</td><td>1</td></tr>
      <tr><td>4</td><td>Schema</td><td><a href="/Archives/edgar/data/1/2/a.xsd">a.xsd</a></td><td>EX-101.SCH</td><td>1</td></tr>
    </table>"""
    assert ff.exhibit_urls(index) == ["https://www.sec.gov/Archives/edgar/data/1/2/ex991.htm"]


# --- Découpage et ancrage ---------------------------------------------------
def test_chunks_lose_nothing_and_respect_the_size():
    text = "\n".join(f"Paragraph {i} " + "x" * (i * 37 % 900) for i in range(200)) + "\n" + "y" * 5000
    chunks = fx.split_chunks(text, size=2000)
    assert all(len(c) <= 2000 for c in chunks) and len(chunks) > 5
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_grounding_accepts_exact_quotes_and_rejects_paraphrases():
    document = "Apple today announced financial results. Services set new all-time revenue\nrecord of $20.8 billion."
    assert fx.grounding_rate(["Services set new all-time revenue record of $20.8 billion."], document) == 1.0
    assert fx.grounding_rate(["“Services set new  all-time revenue record”"], document) == 1.0   # typographie tolérée
    assert fx.grounding_rate(["Services revenue reached a record high this quarter"], document) == 0.0
    assert fx.grounding_rate(["record"], document) == 0.0                                        # trop court
    assert fx.grounding_rate(["Services set new all-time revenue record", "invented sentence here"], document) == 0.5
    assert fx.grounding_rate([], document) == 0.0


# --- Schéma -----------------------------------------------------------------
def test_valid_signals_and_nullable_amounts():
    assert FilingSignals.model_validate(SIGNALS).revenue_musd == 117154.0
    assert FilingSignals.model_validate({**SIGNALS, "revenue_musd": None, "eps_diluted": None}).eps_diluted is None


@pytest.mark.parametrize("override", [
    {"revenue_musd": 117_154_000_000.0},       # dollars au lieu de millions : l'erreur type d'un petit modèle
    {"revenue_musd": -5.0},
    {"eps_diluted": 29_998.0},
    {"guidance_direction": 2},
    {"demand_tone": 3},
    {"legal_risk": -1},
    {"liquidity_risk": 4},
    {"event_type": "BUY_SIGNAL"},
    {"evidence": []},
    {"evidence": ["  "]},
    {"recommendation": "BUY"},                 # le LLM décrit, il ne recommande pas
])
def test_implausible_or_out_of_scope_signals_are_rejected(override):
    with pytest.raises(ValidationError):
        FilingSignals.model_validate({**SIGNALS, **override})


def test_extraction_never_raises_and_records_the_failure(monkeypatch):
    def boom(coroutine):
        coroutine.close()
        raise ConnectionError("LM Studio éteint")

    monkeypatch.setattr(fx, "_run_async", boom)
    monkeypatch.setattr(fx, "company_name", lambda ticker: "Apple Inc.")
    record = fx.extract_signals({"accession": "a", "ticker": "AAPL", "form": "8-K", "filed": "2023-02-02",
                                 "text": "short document"})
    assert record["signals_json"] is None and "ConnectionError" in record["error"]
    assert record["chunks"] == 1 and record["seconds"] >= 0


# --- Features point-in-time -------------------------------------------------
INDEX = pd.bdate_range("2023-01-02", "2023-12-29")


def filing(available: str, form: str = "8-K", **fields) -> dict:
    items = fields.pop("items", "2.02,9.01" if fields.get("event_type", "EARNINGS") == "EARNINGS" else "8.01")
    return {"ticker": "AAPL", "form": form, "available": available, "items": items, **{**SIGNALS, **fields}}


def test_report_features_start_on_the_availability_date_and_expire():
    feats = features_from_signals(pd.DataFrame([filing("2023-02-03", guidance_direction=1)]), INDEX)
    assert feats.loc[:"2023-02-02"].isna().all().all()               # rien avant la publication
    assert feats.loc["2023-02-03", "fil_guidance"] == 1
    assert feats.loc["2023-02-03", "fil_days_since_report"] == 0
    assert feats.loc["2023-05-12", "fil_guidance"] == 1               # 98 jours plus tard : encore valable
    assert pd.isna(feats.loc["2023-05-15", "fil_guidance"])           # 101 jours : périmé
    assert pd.isna(feats.loc["2023-05-15", "fil_days_since_report"])
    assert feats.loc["2023-05-15", "fil_events_30d"] == 0             # couvert, mais aucun événement


def test_weekend_publication_is_usable_on_the_next_session():
    feats = features_from_signals(pd.DataFrame([filing("2023-03-11")]), INDEX)   # samedi
    assert pd.isna(feats.loc["2023-03-10", "fil_demand_tone"])
    assert feats.loc["2023-03-13", "fil_demand_tone"] == 1


def test_reports_are_averaged_and_risks_take_the_maximum():
    signals = pd.DataFrame([
        filing("2023-02-03", guidance_direction=1, legal_risk=0, revenue_growth_yoy_pct=10.0),
        filing("2023-02-06", form="10-Q", event_type="OTHER", guidance_direction=-1, legal_risk=3,
               revenue_growth_yoy_pct=None),
    ])
    row = features_from_signals(signals, INDEX).loc["2023-02-10"]
    assert row["fil_guidance"] == 0 and row["fil_legal_risk"] == 3
    assert row["fil_rev_growth"] == 10.0                                # une valeur absente n'est pas un zéro
    assert row["fil_days_since_report"] == 4


def test_events_are_counted_over_thirty_days():
    signals = pd.DataFrame([
        filing("2023-03-01", event_type="LEGAL_REGULATORY", demand_tone=-1, legal_risk=2),
        filing("2023-03-10", event_type="MANAGEMENT_CHANGE", demand_tone=0, legal_risk=0),
    ])
    feats = features_from_signals(signals, INDEX)
    assert feats.loc["2023-03-15", ["fil_events_30d", "fil_event_tone_30d", "fil_event_risk_30d"]].tolist() == [2, -1, 2]
    assert feats.loc["2023-04-05", "fil_events_30d"] == 1
    assert feats.loc["2023-04-12", "fil_events_30d"] == 0
    assert feats["fil_guidance"].isna().all()                           # aucun rapport : pas de note inventée


def test_future_filings_never_change_past_features():
    past = [filing("2023-02-03"), filing("2023-03-01", event_type="OTHER")]
    future = [filing("2023-06-01", guidance_direction=-1, legal_risk=3), filing("2023-06-02", event_type="M_AND_A")]
    before = features_from_signals(pd.DataFrame(past), INDEX).loc[:"2023-05-31"]
    after = features_from_signals(pd.DataFrame(past + future), INDEX).loc[:"2023-05-31"]
    pd.testing.assert_frame_equal(before, after)


def test_no_filings_means_missing_values():
    feats = features_from_signals(pd.DataFrame(), INDEX)
    assert list(feats.columns) == FILING_FEATURES and feats.isna().all().all()


# --- Évaluation -------------------------------------------------------------
def test_classification_separates_unit_errors_from_wrong_numbers():
    truth = 117_154_000_000.0
    assert study.classify(117_154e6, truth) == "exact"
    assert study.classify(117_200e6, truth) == "exact"                  # arrondi du communiqué
    assert study.classify(117_154e12, truth) == "unit_error"            # dollars pris pour des millions
    assert study.classify(117.154e6, truth) == "unit_error"             # milliards pris pour des millions
    assert study.classify(383_285e6, truth) == "wrong"                  # chiffre annuel au lieu du trimestre
    assert study.classify(None, truth) == "missing"
    assert study.classify(1.88, 1.88, absolute=0.011) == "exact"
    assert study.classify(1.89, 1.88, absolute=0.011) == "exact"
    assert study.classify(0.00188, 1.88, absolute=0.011) == "unit_error"
    assert study.classify(6.11, 1.88, absolute=0.011) == "wrong"


REFERENCE = pd.DataFrame([
    {"ticker": "AAPL", "metric": "revenue", "period_start": "2022-09-25", "period_end": "2022-12-31",
     "value": 117_154e6, "filed": "2023-02-03"},
    {"ticker": "AAPL", "metric": "revenue", "period_start": "2021-09-26", "period_end": "2021-12-25",
     "value": 123_945e6, "filed": "2022-01-28"},
    {"ticker": "AAPL", "metric": "eps_diluted", "period_start": "2022-09-25", "period_end": "2022-12-31",
     "value": 1.88, "filed": "2023-02-03"},
])


def test_reference_is_the_quarter_published_right_after_the_filing():
    assert study.reference_value(REFERENCE, "AAPL", "revenue", "2023-02-02") == 117_154e6   # 8-K la veille du 10-Q
    assert study.reference_value(REFERENCE, "AAPL", "revenue", "2023-06-01") is None
    assert study.reference_value(REFERENCE, "MSFT", "revenue", "2023-02-02") is None


def test_accuracy_report_counts_each_kind_of_error():
    signals = pd.DataFrame([
        {"ticker": "AAPL", "form": "8-K", "items": "2.02,9.01", "filed": "2023-02-02", **SIGNALS},
        {"ticker": "AAPL", "form": "10-Q", "items": "", "filed": "2023-02-03",
         **{**SIGNALS, "revenue_musd": 117.154, "eps_diluted": None}},
        # communiqué de résultats mal étiqueté par le LLM : il reste vérifié, grâce à la rubrique 2.02
        {"ticker": "AAPL", "form": "8-K", "items": "2.02", "filed": "2023-02-02",
         **{**SIGNALS, "event_type": "FINANCING", "revenue_musd": 383_285.0}},
        {"ticker": "AAPL", "form": "8-K", "items": "5.07", "filed": "2023-02-02", **SIGNALS},   # assemblée : ignoré
    ])
    report = study.accuracy_report(signals, REFERENCE)
    assert report["revenue"] == {"verifiable": 3, "exact": 1, "unit_error": 1, "wrong": 1, "missing": 0,
                                 "exact_rate": 1 / 3}
    assert (report["eps"]["exact"], report["eps"]["missing"]) == (2, 1)
    assert report["event_type"] == {"verifiable": 2, "exact": 1, "exact_rate": 0.5}


def test_control_gives_each_company_the_filings_of_another():
    days = pd.bdate_range("2023-01-02", periods=5)
    data = pd.concat([pd.DataFrame({"ticker": t, "fil_guidance": float(i), "rsi_14": 50.0}, index=days)
                      for i, t in enumerate(["A", "B", "C"])])
    swapped = study.swap_companies(data, ["fil_guidance"], seed=0)
    own = data.groupby("ticker")["fil_guidance"].first()
    got = swapped.groupby("ticker")["fil_guidance"].first()
    assert (own != got).all() and sorted(got) == sorted(own)            # personne ne garde ses propres dépôts
    assert swapped["rsi_14"].equals(data["rsi_14"]) and swapped["fil_guidance"].groupby(swapped["ticker"]).nunique().eq(1).all()


def test_cost_report_and_markdown(monkeypatch):
    runs = pd.DataFrame([
        {"form": "8-K", "model": "m", "error": None, "seconds": 100.0, "input_tokens": 3000, "output_tokens": 250,
         "calls": 1, "chars": 11_000, "chunks": 1, "grounding": 1.0},
        {"form": "10-Q", "model": "m", "error": "JSON invalide", "seconds": 300.0, "input_tokens": 9000,
         "output_tokens": 900, "calls": 5, "chars": 20_000, "chunks": 2, "grounding": np.nan},
    ])
    cost = study.cost_report(runs)
    assert (cost["documents"], cost["failures"], cost["calls"]) == (2, 1, 6)
    assert cost["by_form"]["10-Q"]["avg_chunks"] == 2 and cost["grounding"] == 1.0
    page = study.render_markdown({"cost": cost, "accuracy": {}, "predictive": {
        "conclusive": False, "reason": "3 dépôts lus", "coverage": 0.1, "filings_read": 1}})
    assert "Non concluant : 3 dépôts lus" in page and "| 10-Q | 1 | 1 |" in page
    assert "Aucun dépôt" in study.render_markdown({"cost": {"documents": 0}, "accuracy": {}, "predictive": {}})


# --- Lecture par lots -------------------------------------------------------
@pytest.fixture
def batch(monkeypatch, tmp_path):
    """Trois dépôts factices et une base temporaire ; ``extract_signals`` est remplacé par une doublure."""
    import sqlite3

    db = tmp_path / "m.db"
    monkeypatch.setattr(fx, "db_connect", lambda: sqlite3.connect(db))
    filings = pd.DataFrame({
        "accession": ["a", "b", "c"], "ticker": "AAPL", "form": ["8-K", "10-Q", "8-K"],
        "filed": ["2023-02-02", "2023-02-03", "2023-03-10"], "items": ["2.02,9.01", "", "5.07"],
        "text": ["x" * 300, "x" * 900, "x" * 100]})
    monkeypatch.setattr(fx, "load_filings", lambda *args, **kwargs: filings)
    seen: list[str] = []

    def install(error=None):
        def fake(filing):
            seen.append(filing["accession"])
            return {"accession": filing["accession"], "model": fx.MODEL_ID, "prompt_version": fx.PROMPT_VERSION,
                    "signals_json": None if error else "{}", "grounding": None if error else 1.0, "chunks": 1,
                    "calls": 1, "input_tokens": 1, "output_tokens": 1, "seconds": 1.0, "error": error,
                    "created_at": "2026-01-01T00:00:00"}
        monkeypatch.setattr(fx, "extract_signals", fake)

    def stored() -> int:
        with sqlite3.connect(db) as con:
            return con.execute("SELECT COUNT(*) FROM filing_signals").fetchone()[0]

    return install, seen, stored


def test_batch_can_target_earnings_and_start_with_short_documents(batch):
    install, seen, stored = batch
    install()
    assert fx.extract_all(earnings_only=True) == 1 and seen == ["a"]
    seen.clear()
    assert fx.extract_all(cheapest_first=True) == 2 and seen == ["c", "b"]       # « a » est déjà lu
    assert fx.extract_all() == 0 and stored() == 3


def test_batch_stops_on_model_outage_without_recording_failures(batch):
    install, seen, stored = batch
    install(error="ConnectError: Ollama éteint")
    assert fx.extract_all() == 0 and stored() == 0          # rien n'est marqué comme lu : tout sera repris


def test_invalid_json_is_recorded_as_a_real_reading_failure(batch):
    install, seen, stored = batch
    install(error="JSON invalide : demand_tone hors bornes")
    assert fx.extract_all() == 3 and stored() == 3          # échec du modèle sur le fond, pas une panne


def test_batch_respects_the_time_budget(batch, monkeypatch):
    install, seen, stored = batch
    install()
    clock = iter([0.0, 10.0, 4000.0, 8000.0, 12000.0])
    monkeypatch.setattr(fx.time, "monotonic", lambda: next(clock))
    assert fx.extract_all(max_hours=1.0) == 1 and seen == ["a"]
