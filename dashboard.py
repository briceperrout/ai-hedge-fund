"""Tableau de bord du paper-trading : une page HTML autonome, régénérée après chaque cycle.

Sources : ``data/live_state.json`` (portefeuille simulé), table ``trades`` (journal),
``data/live_decisions.jsonl`` (signaux et verdicts du comité), prix du S&P 500 (référence).

Usage :
    python dashboard.py                     # portefeuille live -> data/dashboard.html
    python dashboard.py --run bt-...        # même page pour un backtest (data/reports/<run>/dashboard.html)
    python dashboard.py --open              # ouvre la page dans le navigateur
"""
from __future__ import annotations

import argparse
import html
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_pipeline.fetch_history import db_connect, load_macro, load_prices
from quant_engine.xgboost_model import StrategyConfig, compute_atr, label_trades

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BLOCKED = ("VETO", "ERROR")
OUTCOME_LABELS = {"APPROVED": "Autorisé", "VETO": "Veto", "ERROR": "Bloqué (comité en panne)",
                  "NO_SLOT": "Sans place", "NO_SIZE": "Taille nulle"}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
BOOKS = {"paper": ("Avec comité IA", "live-paper"), "paper_quant": ("Quantitatif seul", "live-quant")}


def _curve(engine: dict[str, Any]) -> pd.Series:
    return pd.Series({pd.Timestamp(d): float(v) for d, v in engine.get("equity_curve", [])}, dtype=float)


def load_live() -> dict[str, Any]:
    """Portefeuille principal (celui du comité s'il existe, sinon le témoin) et, le cas échéant, l'autre."""
    state_path = DATA / "live_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    books = [key for key in BOOKS if key in state]
    main = books[0] if books else "paper"
    engine = state.get(main, {}).get("engine", {})
    label, run_id = BOOKS[main]
    with db_connect() as con:
        trades = pd.read_sql_query("SELECT * FROM trades WHERE mode = 'live' ORDER BY exit_date, id", con)
    decisions_path = DATA / "live_decisions.jsonl"
    decisions = (pd.DataFrame([json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()])
                 if decisions_path.exists() else pd.DataFrame())
    others = []
    for key in books[1:]:
        other = state[key]["engine"]
        others.append({"label": BOOKS[key][0], "curve": _curve(other), "initial": float(other["initial_capital"]),
                       "trades": trades[trades["run_id"] == BOOKS[key][1]]})
    return {
        "title": "Paper-trading en continu", "subtitle": f"Portefeuille simulé, argent fictif — {label.lower()}",
        "label": label, "others": others,
        "initial": float(engine.get("initial_capital", 0.0)), "cash": float(engine.get("cash", 0.0)),
        "curve": _curve(engine),
        "positions": engine.get("positions", {}), "pending": engine.get("pending", {}),
        "last_price": engine.get("last_price", {}), "emergency": bool(engine.get("emergency_exit", False)),
        "trades": trades[trades["run_id"] == run_id], "decisions": decisions,
    }


def load_backtest(run_id: str) -> dict[str, Any]:
    folder = DATA / "reports" / run_id
    report = json.loads((folder / "report.json").read_text())
    curve = pd.read_csv(folder / "equity.csv", parse_dates=["date"]).set_index("date")["equity"]
    read = lambda name: pd.read_csv(folder / name) if (folder / name).exists() else pd.DataFrame()  # noqa: E731
    return {
        "title": f"Backtest {run_id}",
        "subtitle": "Simulation historique, " + ("comité IA actif" if report.get("llm") else "moteur quantitatif seul"),
        "initial": float(report["capital"]["initial"]), "cash": float(report["capital"]["final"]),
        "curve": curve.astype(float), "positions": {}, "pending": {}, "last_price": {}, "emergency": False,
        "trades": read("trades.csv"), "decisions": read("decisions.csv"),
    }


def add_hypothetical_outcomes(decisions: pd.DataFrame) -> pd.DataFrame:
    """Complète chaque signal avec l'issue qu'aurait eue le trade, dès qu'elle est connue.

    En live, cette issue n'existe pas le soir du verdict : on la recalcule ici avec les mêmes
    barrières que l'entraînement. C'est ce qui permet de juger les vetos après coup.
    """
    if not len(decisions) or "ticker" not in decisions:
        return decisions
    decisions = decisions.copy()
    if "hypothetical_ret" not in decisions:
        decisions["hypothetical_ret"] = np.nan
    cfg = StrategyConfig.from_env()
    for ticker, group in decisions[decisions["hypothetical_ret"].isna()].groupby("ticker"):
        prices = load_prices(ticker)
        if len(prices) < 30:
            continue
        labels = label_trades(prices, compute_atr(prices), cfg)["trade_ret"]
        for idx, day in group["date"].items():
            day = pd.Timestamp(day)
            if day in labels.index and pd.notna(labels.loc[day]):
                decisions.loc[idx, "hypothetical_ret"] = float(labels.loc[day])
    return decisions


# ---------------------------------------------------------------------------
# Indicateurs
# ---------------------------------------------------------------------------
def compute_kpis(data: dict[str, Any]) -> dict[str, Any]:
    curve, trades, initial = data["curve"], data["trades"], data["initial"]
    kpis: dict[str, Any] = {"sessions": int(len(curve)), "trades": int(len(trades))}
    if len(curve):
        returns = curve.pct_change().dropna()
        kpis.update(
            equity=float(curve.iloc[-1]), total_return=float(curve.iloc[-1] / initial - 1) if initial else 0.0,
            max_drawdown=float((curve / curve.cummax() - 1).min()),
            sharpe=float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 20 and returns.std() > 0 else None,
            start=curve.index[0], end=curve.index[-1])
    if len(trades):
        kpis.update(win_rate=float((trades["pnl"] > 0).mean()), pnl=float(trades["pnl"].sum()))
    return kpis


def veto_scoreboard(decisions: pd.DataFrame) -> Optional[dict[str, Any]]:
    if not len(decisions) or "outcome" not in decisions:
        return None
    consulted = decisions[decisions["outcome"].isin(("APPROVED", "NO_SIZE") + BLOCKED)]
    blocked = consulted[consulted["outcome"].isin(BLOCKED)]
    approved = consulted[~consulted["outcome"].isin(BLOCKED)]
    board: dict[str, Any] = {"signals": int(len(consulted)), "blocked": int(len(blocked)), "approved": int(len(approved))}
    for name, group in (("blocked", blocked), ("approved", approved)):
        known = group["hypothetical_ret"].dropna() if "hypothetical_ret" in group else pd.Series(dtype=float)
        board[f"{name}_known"] = int(len(known))
        board[f"{name}_avg"] = float(known.mean()) if len(known) else None
    return board


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: Optional[float], digits: int = 2) -> str:
    return "n/d" if value is None or pd.isna(value) else f"{value:+.{digits}%}".replace("-", "−")


def usd(value: float, signed: bool = False) -> str:
    text = f"{value:+,.0f}" if signed else f"{value:,.0f}"
    return text.replace(",", " ").replace("-", "−") + " $"


def nice_ticks(low: float, high: float, count: int = 4) -> list[float]:
    span = max(high - low, 1e-9)
    step = 10 ** np.floor(np.log10(span / count))
    step = next(step * m for m in (1, 2, 2.5, 5, 10) if span / (step * m) <= count)
    start = np.ceil(low / step) * step
    return [float(t) for t in np.arange(start, high + step * 0.01, step)]


def equity_chart(curve: pd.Series, label: str = "Portefeuille", others: Optional[list[dict[str, Any]]] = None) -> str:
    """Portefeuille(s) et S&P 500, en base 100 à leur premier jour affiché (un seul axe)."""
    if len(curve) < 2:
        return '<p class="empty">La courbe apparaîtra après deux séances traitées.</p>'
    portfolio = curve / curve.iloc[0] * 100
    spy = load_macro().get("spy", pd.Series(dtype=float)).reindex(curve.index).ffill().bfill()
    series = [(label, "s1", portfolio)]
    for other in (others or [])[:1]:
        aligned = other["curve"].reindex(curve.index)
        if aligned.notna().sum() >= 2:
            series.append((other["label"], "s3", aligned / aligned.dropna().iloc[0] * 100))
    if spy.notna().all() and len(spy):
        series.append(("S&P 500", "s2", spy / spy.iloc[0] * 100))

    width, height, left, right, top, bottom = 800, 300, 44, 96, 14, 28
    lo = min(s.min() for _, _, s in series)
    hi = max(s.max() for _, _, s in series)  # min/max ignorent les valeurs manquantes
    pad = max((hi - lo) * 0.08, 0.5)
    lo, hi = lo - pad, hi + pad
    n = len(curve)
    x = lambda i: left + (width - left - right) * i / (n - 1)          # noqa: E731
    y = lambda v: top + (height - top - bottom) * (1 - (v - lo) / (hi - lo))  # noqa: E731

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Valeur du portefeuille et du S&amp;P 500, '
             f'base 100 au {curve.index[0]:%d/%m/%Y}" id="chart">']
    for tick in nice_ticks(lo, hi):
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
                     f'<text class="axis" x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>')
    if lo < 100 < hi:
        parts.append(f'<line class="base" x1="{left}" x2="{width - right}" y1="{y(100):.1f}" y2="{y(100):.1f}"/>')
    months = pd.Series(range(n), index=curve.index).groupby(curve.index.to_period("M")).first()
    stride = max(1, int(np.ceil(len(months) / 8)))
    for period, i in list(months.items())[::stride]:
        label = f"{period.start_time:%b %y}" if len(months) > 1 else f"{curve.index[i]:%d/%m}"
        parts.append(f'<text class="axis" x="{x(i):.1f}" y="{height - 8}" text-anchor="middle">{label}</text>')

    ends = []
    for name, cls, values in series:
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values) if pd.notna(v))
        parts.append(f'<polyline class="line {cls}" points="{points}"/>')
        last_value = float(values.dropna().iloc[-1])
        ends.append([name, cls, last_value, y(last_value)])
    ends.sort(key=lambda e: e[3])
    for a, b in zip(ends, ends[1:]):  # écarte les étiquettes de fin de courbe qui se chevauchent
        if b[3] - a[3] < 30:
            middle = (a[3] + b[3]) / 2
            a[3], b[3] = middle - 15, middle + 15
    for name, cls, last, pos in ends:
        parts.append(f'<circle class="dot {cls}" cx="{x(n - 1):.1f}" cy="{y(last):.1f}" r="4"/>'
                     f'<text class="endlabel" x="{x(n - 1) + 10:.1f}" y="{pos - 2:.1f}">{esc(name)}</text>'
                     f'<text class="endvalue" x="{x(n - 1) + 10:.1f}" y="{pos + 12:.1f}">{last:.1f}</text>')
    parts.append(f'<line id="cross" class="cross" y1="{top}" y2="{height - bottom}" visibility="hidden"/>')
    for _, cls, _ in series:
        parts.append(f'<circle class="dot hover {cls}" r="4" visibility="hidden"/>')
    parts.append(f'<rect id="hit" x="{left}" y="{top}" width="{width - left - right}" '
                 f'height="{height - top - bottom}" fill="transparent"/></svg>')

    payload = {"dates": [d.strftime("%d/%m/%Y") for d in curve.index],
               "series": [{"name": name, "cls": cls,
                           "values": [round(float(v), 2) if pd.notna(v) else None for v in values],
                           "y": [round(y(v), 1) if pd.notna(v) else None for v in values]}
                          for name, cls, values in series],
               "x": [round(x(i), 1) for i in range(n)], "width": width}
    legend = "".join(f'<span><i class="swatch {cls}"></i>{esc(name)}</span>' for name, cls, _ in series)
    table = "".join(
        f"<tr><td>{d:%d/%m/%Y}</td>" + "".join(f'<td class="num">{s.iloc[i]:.1f}</td>' if pd.notna(s.iloc[i]) else '<td class="num">—</td>'
                                         for _, _, s in series) + "</tr>"
        for i, d in list(enumerate(curve.index))[-30:][::-1])
    head = "".join(f'<th class="num">{esc(name)}</th>' for name, _, _ in series)
    return (f'<div class="legend">{legend}</div><div class="chartwrap">{"".join(parts)}<div id="tip" hidden></div></div>'
            f'<details><summary>Voir les 30 dernières valeurs</summary><div class="scroll"><table><thead><tr><th>Séance</th>'
            f'{head}</tr></thead><tbody>{table}</tbody></table></div></details>'
            f'<script id="chart-data" type="application/json">{json.dumps(payload)}</script>')


def table(headers: list[tuple[str, bool]], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{esc(empty)}</p>'
    head = "".join(f'<th{" class=num" if numeric else ""}>{esc(name)}</th>' for name, numeric in headers)
    body = "".join("<tr>" + "".join(f'<td{" class=num" if headers[i][1] else ""}>{cell}</td>'
                                    for i, cell in enumerate(row)) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(data: dict[str, Any]) -> str:
    kpis = compute_kpis(data)
    decisions = add_hypothetical_outcomes(data["decisions"])
    board = veto_scoreboard(decisions)
    trades = data["trades"]

    tiles = [
        ("Capital", usd(kpis["equity"]) if "equity" in kpis else "n/d", f"départ {usd(data['initial'])}"),
        ("Rendement", pct(kpis.get("total_return")), f"{kpis['sessions']} séances"),
        ("Ratio de Sharpe", f"{kpis['sharpe']:.2f}".replace("-", "−") if kpis.get("sharpe") is not None else "n/d",
         "annualisé" if kpis.get("sharpe") is not None else "après 20 séances"),
        ("Perte maximale", pct(kpis.get("max_drawdown")), "depuis un sommet"),
        ("Trades clôturés", str(kpis["trades"]),
         f"{kpis['win_rate']:.0%} gagnants" if "win_rate" in kpis else "aucun pour l'instant"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="label">{esc(a)}</div><div class="value">{esc(b)}</div>'
                         f'<div class="hint">{esc(c)}</div></div>' for a, b, c in tiles)

    positions = [[esc(t), str(p["shares"]), f"{p['entry_price']:.2f}", f"{data['last_price'].get(t, p['entry_price']):.2f}",
                  pct(data["last_price"].get(t, p["entry_price"]) / p["entry_price"] - 1), f"{p['stop_loss']:.2f}",
                  f"{p['take_profit']:.2f}", esc(p["entry_date"])] for t, p in data["positions"].items()]
    pending = ", ".join(f"{esc(t)} ×{o['plan']['shares']}" for t, o in data["pending"].items())

    trade_rows = [[esc(r["ticker"]), esc(r["entry_date"]), esc(r["exit_date"]), esc(r["exit_reason"]),
                   usd(r["pnl"], signed=True), pct(r["pnl_pct"])] for _, r in trades.iloc[::-1].head(25).iterrows()]

    decision_rows = []
    if len(decisions):
        for _, r in decisions.iloc[::-1].head(25).iterrows():
            later = r.get("hypothetical_ret")
            decision_rows.append([
                esc(r["date"]), esc(r["ticker"]), f"{r['probability']:.0%}",
                esc(OUTCOME_LABELS.get(r["outcome"], r["outcome"])),
                pct(later) if pd.notna(later) else "en cours",
                esc(r["justification"]) if pd.notna(r.get("justification")) else "—"])

    if board and board["signals"]:
        approved_avg, blocked_avg = board["approved_avg"], board["blocked_avg"]
        sentence = (f"Le comité a étudié <b>{board['signals']}</b> signaux et en a bloqué <b>{board['blocked']}</b>. ")
        if blocked_avg is not None and approved_avg is not None:
            sentence += (f"Les signaux bloqués auraient rapporté <b>{pct(blocked_avg)}</b> en moyenne "
                         f"({board['blocked_known']} issues connues), les signaux autorisés <b>{pct(approved_avg)}</b> "
                         f"({board['approved_known']}). Un veto utile se reconnaît à un premier chiffre nettement "
                         "inférieur au second ; quelques dizaines de vetos sont nécessaires pour conclure.")
        elif not board["blocked"]:
            sentence += "Sans veto, sa valeur ne peut pas encore être mesurée."
        else:
            sentence += "L'issue des signaux bloqués n'est pas encore connue."
        veto_html = f"<p>{sentence}</p>"
    else:
        veto_html = '<p class="empty">Aucun signal soumis au comité pour l\'instant.</p>'

    compare_html = ""
    if data.get("others"):
        rows = []
        for name, book in [(data.get("label", "Portefeuille"), data)] + [(o["label"], o) for o in data["others"]]:
            k = compute_kpis({"curve": book["curve"], "trades": book["trades"], "initial": book["initial"]})
            rows.append([esc(name), usd(k["equity"]) if "equity" in k else "n/d", pct(k.get("total_return")),
                         pct(k.get("max_drawdown")), str(k["trades"]),
                         f"{k['win_rate']:.0%}" if "win_rate" in k else "n/d"])
        compare_html = ("<section><h2>Comité IA contre quantitatif seul, sur les mêmes signaux</h2>"
                        + table([("Portefeuille", False), ("Capital", True), ("Rendement", True), ("Perte maximale", True),
                                 ("Trades", True), ("Gagnants", True)], rows, "")
                        + '<p class="hint">Le comité ne peut que bloquer des trades : son apport se lit dans l\'écart '
                          "entre ces deux lignes, mesuré en temps réel.</p></section>")

    period = (f"du {kpis['start']:%d/%m/%Y} au {kpis['end']:%d/%m/%Y}" if "start" in kpis
              else "aucune séance traitée pour l'instant")
    alert = ('<p class="alert">⚠ Alerte systémique du CRO : toutes les positions seront coupées à la prochaine ouverture.</p>'
             if data["emergency"] else "")
    return PAGE.format(
        title=esc(data["title"]), subtitle=esc(data["subtitle"]), period=period, alert=alert, tiles=tiles_html,
        chart=equity_chart(data["curve"], data.get("label", "Portefeuille"), data.get("others")),
        compare=compare_html,
        positions=table([("Valeur", False), ("Actions", True), ("Entrée", True), ("Dernier", True), ("Latent", True),
                         ("Stop", True), ("Objectif", True), ("Depuis", False)], positions, "Aucune position ouverte."),
        pending=f"<p>Ordres pour la prochaine ouverture : {pending}</p>" if pending else "",
        veto=veto_html,
        decisions=table([("Date", False), ("Valeur", False), ("Probabilité", True), ("Verdict", False),
                         ("Issue du trade", True), ("Justification du CRO", False)], decision_rows,
                        "Aucun signal enregistré."),
        trades=table([("Valeur", False), ("Entrée", False), ("Sortie", False), ("Motif", False), ("P&L", True),
                      ("Rendement", True)], trade_rows, "Aucun trade clôturé."),
        generated=f"{datetime.now():%d/%m/%Y à %H:%M}")


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; --page:#f4f3f0; --surface:#fcfcfb; --border:#e3e1da; --text:#0b0b0b; --text2:#52514e;
  --muted:#85837c; --grid:#e9e7e1; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; --page:#111110; --surface:#1a1a19; --border:#33322f;
  --text:#f0efec; --text2:#c3c2b7; --muted:#8f8d85; --grid:#2a2927; --s1:#3987e5; --s2:#d95926; --s3:#199e70; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--text); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:28px 16px 48px; }}
h1 {{ font-size:24px; margin:0; letter-spacing:-0.01em; }}
h2 {{ font-size:15px; margin:0 0 12px; }}
.sub {{ color:var(--text2); margin:4px 0 20px; }}
.alert {{ border:1px solid var(--s2); border-radius:8px; padding:10px 14px; background:var(--surface); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:12px; }}
.tile, section {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }}
section {{ margin-top:12px; }}
.label {{ color:var(--text2); font-size:13px; }}
.value {{ font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }}
.hint, .empty, footer {{ color:var(--muted); font-size:13px; }}
.legend {{ display:flex; gap:16px; font-size:13px; color:var(--text2); margin-bottom:4px; }}
.swatch {{ display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; }}
.swatch.s1 {{ background:var(--s1); }} .swatch.s2 {{ background:var(--s2); }} .swatch.s3 {{ background:var(--s3); }}
.chartwrap {{ position:relative; }}
svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .base {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3; }}
.axis {{ fill:var(--muted); font-size:11px; }}
.line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.line.s1, .cross {{ stroke:var(--s1); }} .line.s2 {{ stroke:var(--s2); }} .line.s3 {{ stroke:var(--s3); }}
.dot {{ stroke:var(--surface); stroke-width:2; }} .dot.s1 {{ fill:var(--s1); }} .dot.s2 {{ fill:var(--s2); }} .dot.s3 {{ fill:var(--s3); }}
.cross {{ stroke:var(--muted); stroke-width:1; }}
.endlabel {{ fill:var(--text2); font-size:11px; }} .endvalue {{ fill:var(--text); font-size:12px; font-weight:600; }}
#tip {{ position:absolute; top:6px; pointer-events:none; background:var(--surface); border:1px solid var(--border);
  border-radius:6px; padding:6px 10px; font-size:12px; white-space:nowrap; box-shadow:0 2px 8px rgba(0,0,0,.12); }}
#tip b {{ font-variant-numeric:tabular-nums; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th {{ text-align:left; color:var(--text2); font-weight:500; border-bottom:1px solid var(--border); padding:6px 10px 6px 0; white-space:nowrap; }}
td {{ border-bottom:1px solid var(--grid); padding:6px 10px 6px 0; vertical-align:top; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
details {{ margin-top:8px; font-size:13px; color:var(--text2); }}
footer {{ margin-top:20px; }}
</style></head><body><main>
<h1>{title}</h1>
<p class="sub">{subtitle} — {period}</p>
{alert}
<div class="tiles">{tiles}</div>
<section><h2>Valeur du portefeuille et du S&amp;P 500, base 100 au premier jour</h2>{chart}</section>
{compare}
<section><h2>Positions ouvertes</h2>{positions}{pending}</section>
<section><h2>Le veto du CRO est-il utile ?</h2>{veto}{decisions}</section>
<section><h2>Derniers trades clôturés</h2>{trades}</section>
<footer>Page générée le {generated}. Projet de recherche : aucun ordre réel, aucune recommandation d'investissement.</footer>
</main>
<script>
(function () {{
  var el = document.getElementById('chart-data'); if (!el) return;
  var d = JSON.parse(el.textContent), svg = document.getElementById('chart'), hit = document.getElementById('hit');
  var cross = document.getElementById('cross'), tip = document.getElementById('tip');
  var dots = svg.querySelectorAll('.dot.hover');
  function move(evt) {{
    var box = svg.getBoundingClientRect(), px = (evt.clientX - box.left) * d.width / box.width, best = 0;
    for (var i = 1; i < d.x.length; i++) if (Math.abs(d.x[i] - px) < Math.abs(d.x[best] - px)) best = i;
    cross.setAttribute('x1', d.x[best]); cross.setAttribute('x2', d.x[best]); cross.setAttribute('visibility', 'visible');
    var rows = '';
    d.series.forEach(function (s, k) {{
      if (s.values[best] === null) {{ dots[k].setAttribute('visibility', 'hidden'); return; }}
      dots[k].setAttribute('cx', d.x[best]); dots[k].setAttribute('cy', s.y[best]); dots[k].setAttribute('visibility', 'visible');
      rows += '<div><i class="swatch ' + s.cls + '"></i>' + s.name + ' <b>' + s.values[best].toFixed(1) + '</b></div>';
    }});
    tip.innerHTML = '<div>' + d.dates[best] + '</div>' + rows; tip.hidden = false;
    var left = d.x[best] * box.width / d.width;
    tip.style.left = (left > box.width / 2 ? left - tip.offsetWidth - 12 : left + 12) + 'px';
  }}
  function leave() {{
    cross.setAttribute('visibility', 'hidden'); tip.hidden = true;
    dots.forEach(function (c) {{ c.setAttribute('visibility', 'hidden'); }});
  }}
  hit.addEventListener('pointermove', move); hit.addEventListener('pointerleave', leave);
}})();
</script></body></html>
"""


def build_dashboard(run_id: Optional[str] = None) -> Path:
    data = load_backtest(run_id) if run_id else load_live()
    target = (DATA / "reports" / run_id / "dashboard.html") if run_id else (DATA / "dashboard.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(data), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère le tableau de bord HTML.")
    parser.add_argument("--run", help="Identifiant d'un backtest (dossier de data/reports/).")
    parser.add_argument("--open", action="store_true", help="Ouvrir la page dans le navigateur.")
    args = parser.parse_args()
    path = build_dashboard(args.run)
    print(f"Tableau de bord : {path}")
    if args.open:
        webbrowser.open(path.as_uri())
