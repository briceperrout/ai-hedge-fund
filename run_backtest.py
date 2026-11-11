"""Point d'entrée de la simulation historique (Walk-Forward).

Chaque séance est rejouée dans l'ordre, avec uniquement l'information disponible ce jour-là :

    ouverture  : exécution des ordres validés la veille (et arrêt d'urgence éventuel)
    séance     : stop-loss / objectif / durée max, gérés par Python
    clôture    : XGBoost scanne l'univers ; probabilité >= seuil  ->  dossier (RAG)  ->
                 débat Bull/Bear  ->  veto du CRO  ->  taille de position (Kelly)
    nuit       : autopsie des trades perdants (RAG d'expérience)

Le modèle est ré-entraîné périodiquement sur les seuls trades dont l'issue est déjà connue.

Usage :
    python run_backtest.py                              # année 2023, univers du .env
    python run_backtest.py --no-llm                     # moteur quantitatif seul (référence)
    python run_backtest.py --start 2023-03-01 --end 2023-04-30 --tickers AAPL JPM
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_pipeline.fetch_history import TICKERS, db_connect, load_macro, sync_all, sync_dark_pool
from quant_engine.xgboost_model import (
    StrategyConfig, TradePlan, build_dataset, plan_trade, predict_proba, train_model, training_rows,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

log = logging.getLogger("backtest")

REPORT_DIR = ROOT / "data" / "reports"
VETO_COOLDOWN_DAYS = int(os.getenv("VETO_COOLDOWN_DAYS", "5"))
ISO = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Moteur d'exécution (partagé avec run_live.py) — 100 % Python, aucun LLM
# ---------------------------------------------------------------------------
@dataclass
class Position:
    ticker: str
    shares: int
    signal_date: str
    entry_date: str
    entry_price: float
    stop_loss: float
    take_profit: float
    probability: float
    kelly_fraction: float
    allocation_pct: float
    bars_held: int = 0
    context: dict = field(default_factory=dict)


@dataclass
class PendingOrder:
    plan: TradePlan
    signal_date: str
    context: dict = field(default_factory=dict)


class ExecutionEngine:
    """Portefeuille simulé : exécute les ordres, place et surveille stop-loss et objectifs."""

    def __init__(self, cfg: StrategyConfig, capital: float, run_id: str, mode: str):
        self.cfg, self.run_id, self.mode = cfg, run_id, mode
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, PendingOrder] = {}
        self.cooldown: dict[str, str] = {}      # ticker -> date jusqu'à laquelle le comité n'est pas resollicité
        self.emergency_exit = False             # armé par un veto systémique du CRO
        self.last_price: dict[str, float] = {}
        self.closed: list[dict[str, Any]] = []
        self.equity_curve: list[tuple[str, float]] = []

    # -- valorisation -------------------------------------------------------
    def equity(self) -> float:
        return self.cash + sum(p.shares * self.last_price.get(t, p.entry_price) for t, p in self.positions.items())

    def buying_power(self) -> float:
        return self.cash - sum(o.plan.shares * o.plan.reference_price for o in self.pending.values())

    def free_slots(self) -> int:
        return self.cfg.max_open_positions - len(self.positions) - len(self.pending)

    def mark(self, day: pd.Timestamp, bars: dict[str, Any]) -> float:
        for ticker, bar in bars.items():
            self.last_price[ticker] = float(bar["close"])
        value = self.equity()
        self.equity_curve.append((day.strftime(ISO), value))
        return value

    # -- ouverture ----------------------------------------------------------
    def on_open(self, day: pd.Timestamp, bars: dict[str, Any]) -> list[dict[str, Any]]:
        closed = []
        if self.emergency_exit:
            for ticker in list(self.positions):
                if ticker in bars:
                    closed.append(self._close(ticker, float(bars[ticker]["open"]), day, "ARRÊT D'URGENCE (veto systémique)"))
            self.pending.clear()
            self.emergency_exit = False
        for ticker, order in list(self.pending.items()):
            bar = bars.get(ticker)
            if bar is None:  # pas de cotation aujourd'hui : l'ordre attend
                continue
            del self.pending[ticker]
            open_price = float(bar["open"])
            fill = open_price * (1 + self.cfg.cost_bps / 1e4)
            shares = min(order.plan.shares, int(self.cash // fill))
            if shares <= 0:
                continue
            stop, target = order.plan.levels(open_price)
            self.cash -= shares * fill
            self.positions[ticker] = Position(
                ticker=ticker, shares=shares, signal_date=order.signal_date, entry_date=day.strftime(ISO),
                entry_price=fill, stop_loss=stop, take_profit=target, probability=order.plan.probability,
                kelly_fraction=order.plan.kelly_full, allocation_pct=order.plan.allocation_pct, context=order.context,
            )
            log.info("[%s] ACHAT %s : %d actions à %.2f | stop %.2f | objectif %.2f | Kelly %.0f%% -> %.1f%% du capital",
                     day.date(), ticker, shares, fill, stop, target, order.plan.kelly_full * 100,
                     order.plan.allocation_pct * 100)
        return closed

    # -- séance -------------------------------------------------------------
    def on_session(self, day: pd.Timestamp, bars: dict[str, Any]) -> list[dict[str, Any]]:
        closed = []
        for ticker, pos in list(self.positions.items()):
            bar = bars.get(ticker)
            if bar is None:
                continue
            pos.bars_held += 1
            if bar["low"] <= pos.stop_loss:  # stop testé en premier (hypothèse défavorable, comme à l'entraînement)
                closed.append(self._close(ticker, min(float(bar["open"]), pos.stop_loss), day, "STOP-LOSS"))
            elif bar["high"] >= pos.take_profit:
                closed.append(self._close(ticker, max(float(bar["open"]), pos.take_profit), day, "OBJECTIF ATTEINT"))
            elif pos.bars_held >= self.cfg.horizon:
                closed.append(self._close(ticker, float(bar["close"]), day, "DURÉE MAX"))
        return closed

    def liquidate(self, day: pd.Timestamp, reason: str) -> list[dict[str, Any]]:
        return [self._close(t, self.last_price.get(t, p.entry_price), day, reason) for t, p in list(self.positions.items())]

    def _close(self, ticker: str, price: float, day: pd.Timestamp, reason: str) -> dict[str, Any]:
        pos = self.positions.pop(ticker)
        fill = price * (1 - self.cfg.cost_bps / 1e4)
        self.cash += pos.shares * fill
        trade = {
            **{k: v for k, v in asdict(pos).items() if k != "bars_held"},
            "run_id": self.run_id, "mode": self.mode, "bars_held": pos.bars_held,
            "exit_date": day.strftime(ISO), "exit_price": fill, "exit_reason": reason,
            "pnl": (fill - pos.entry_price) * pos.shares, "pnl_pct": fill / pos.entry_price - 1,
        }
        trade["id"] = _journal_trade(trade)
        self.closed.append(trade)
        log.info("[%s] SORTIE %s (%s) à %.2f : %+.2f USD (%+.2f%%)", day.date(), ticker, reason, fill,
                 trade["pnl"], trade["pnl_pct"] * 100)
        return trade

    # -- persistance (mode live) --------------------------------------------
    def to_state(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "initial_capital": self.initial_capital, "cash": self.cash,
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "pending": {t: {"plan": o.plan.to_dict(), "signal_date": o.signal_date, "context": o.context}
                        for t, o in self.pending.items()},
            "cooldown": self.cooldown, "emergency_exit": self.emergency_exit,
            "last_price": self.last_price, "equity_curve": self.equity_curve,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], cfg: StrategyConfig, mode: str) -> "ExecutionEngine":
        engine = cls(cfg, state["initial_capital"], state["run_id"], mode)
        engine.cash = state["cash"]
        engine.positions = {t: Position(**p) for t, p in state["positions"].items()}
        engine.pending = {t: PendingOrder(TradePlan(**o["plan"]), o["signal_date"], o["context"])
                          for t, o in state["pending"].items()}
        engine.cooldown = state.get("cooldown", {})
        engine.emergency_exit = state.get("emergency_exit", False)
        engine.last_price = state.get("last_price", {})
        engine.equity_curve = [tuple(row) for row in state.get("equity_curve", [])]
        return engine


def _journal_trade(trade: dict[str, Any]) -> int:
    """Écrit le trade clôturé dans le journal SQL."""
    with db_connect() as con:
        cur = con.execute(
            "INSERT INTO trades (run_id, mode, ticker, signal_date, entry_date, entry_price, shares, stop_loss, "
            "take_profit, probability, kelly_fraction, allocation_pct, exit_date, exit_price, exit_reason, pnl, "
            "pnl_pct, context_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade["run_id"], trade["mode"], trade["ticker"], trade["signal_date"], trade["entry_date"],
             trade["entry_price"], trade["shares"], trade["stop_loss"], trade["take_profit"], trade["probability"],
             trade["kelly_fraction"], trade["allocation_pct"], trade["exit_date"], trade["exit_price"],
             trade["exit_reason"], trade["pnl"], trade["pnl_pct"], json.dumps(trade["context"], ensure_ascii=False)),
        )
        con.commit()
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Chaîne de décision : XGBoost -> dossier -> comité -> Kelly
# ---------------------------------------------------------------------------
def evaluate_signals(day: pd.Timestamp, rows: pd.DataFrame, model, features: list[str], cfg: StrategyConfig, *,
                     prices: dict[str, pd.DataFrame], macro: pd.DataFrame, held: set[str], cooldown: dict[str, str],
                     equity: float, cash: float, slots: int, mode: str, scope: Optional[str],
                     use_llm: bool, use_memory: bool, decisions: list[dict[str, Any]]) -> tuple[list[PendingOrder], bool]:
    """Scanne l'univers à la clôture de ``day``.

    Retourne (ordres autorisés à exécuter à la prochaine ouverture, alerte systémique du CRO).
    """
    day_iso = day.strftime(ISO)
    scored = rows.assign(probability=predict_proba(model, features, rows)).sort_values("probability", ascending=False)
    approved: list[PendingOrder] = []
    for _, row in scored.iterrows():
        probability, ticker = float(row["probability"]), row["ticker"]
        if probability < cfg.threshold:
            break
        if ticker in held or cooldown.get(ticker, "") >= day_iso:
            continue
        record = {"date": day_iso, "ticker": ticker, "probability": round(probability, 4),
                  "hypothetical_ret": None if pd.isna(row.get("trade_ret")) else float(row["trade_ret"]),
                  "hypothetical_win": None if pd.isna(row.get("target")) else int(row["target"])}
        if len(approved) >= slots:
            decisions.append({**record, "outcome": "NO_SLOT"})
            continue

        from data_pipeline.market_snapshot import build_investment_dossier  # import tardif : charge les embeddings

        dossier = build_investment_dossier(ticker, day, prices[ticker].loc[:day], macro.loc[:day],
                                           mode=mode, scope=scope, use_memory=use_memory)
        context = {k: dossier[k] for k in ("technical", "macro", "fundamentals", "news", "experience")}
        log.info("[%s] SIGNAL %s : probabilité %.1f%%", day.date(), ticker, probability * 100)

        if use_llm:
            from llm_agents.graph import run_committee

            verdict = run_committee({"statistical_action": "BUY", "win_probability": probability}, dossier)
            decision = verdict["decision"]
            context.update(bull_thesis=verdict["bull_thesis"], bear_thesis=verdict["bear_thesis"],
                           cro_justification=decision.justification, risk_level=decision.risk_level,
                           key_risks=decision.key_risks)
            record.update(risk_level=decision.risk_level, justification=decision.justification,
                          committee_error=verdict["error"])
            if decision.veto_activated:
                cooldown[ticker] = (day + pd.offsets.BDay(VETO_COOLDOWN_DAYS)).strftime(ISO)
                outcome = "ERROR" if verdict["error"] and not decision.systemic_risk_detected else "VETO"
                decisions.append({**record, "outcome": outcome, "systemic": decision.systemic_risk_detected})
                log.info("[%s] VETO du CRO sur %s (%s) : %s", day.date(), ticker, decision.risk_level, decision.justification)
                if decision.systemic_risk_detected:
                    return approved, True
                continue
            log.info("[%s] FEU VERT du CRO sur %s (%s) : %s", day.date(), ticker, decision.risk_level, decision.justification)

        plan = plan_trade(ticker, equity, cash, float(row["close"]), float(row["atr"]), probability, cfg)
        if plan is None:
            decisions.append({**record, "outcome": "NO_SIZE"})
            continue
        cash -= plan.shares * plan.reference_price
        approved.append(PendingOrder(plan, day_iso, context))
        decisions.append({**record, "outcome": "APPROVED", "allocation_pct": round(plan.allocation_pct, 4)})
    return approved, False


def night_autopsy(closed_today: list[dict[str, Any]], prices: dict[str, pd.DataFrame], macro: pd.DataFrame, *,
                  mode: str, scope: str, use_memory: bool) -> int:
    """Agent d'Autopsie : chaque trade perdant devient une leçon vectorisée dans le journal."""
    from data_pipeline.market_snapshot import holding_period_review, record_lesson
    from llm_agents.graph import run_autopsy

    lessons = 0
    for trade in closed_today:
        if trade["pnl"] >= 0:
            continue
        exit_day = pd.Timestamp(trade["exit_date"])
        trade["post_mortem"] = holding_period_review(trade, prices[trade["ticker"]].loc[:exit_day],
                                                     macro.loc[:exit_day], mode=mode, use_memory=use_memory)
        report = run_autopsy(trade)
        if report is None:
            continue
        record_lesson(trade, report.model_dump(), scope)
        with db_connect() as con:
            con.execute("UPDATE trades SET autopsy_done = 1 WHERE id = ?", (trade["id"],))
            con.commit()
        lessons += 1
        log.info("[%s] AUTOPSIE %s (%s) : %s", exit_day.date(), trade["ticker"], report.error_category, report.lesson)
    return lessons


# ---------------------------------------------------------------------------
# Mesure de performance
# ---------------------------------------------------------------------------
def performance_report(engine: ExecutionEngine, decisions: list[dict[str, Any]], macro: pd.DataFrame,
                       prices: dict[str, pd.DataFrame]) -> dict[str, Any]:
    curve = pd.Series({pd.Timestamp(d): v for d, v in engine.equity_curve})
    returns = curve.pct_change().dropna()
    trades = pd.DataFrame(engine.closed)
    start, end = curve.index[0], curve.index[-1]
    years = max((end - start).days / 365.25, 1 / 365.25)
    total_return = curve.iloc[-1] / engine.initial_capital - 1

    report: dict[str, Any] = {
        "period": {"start": start.strftime(ISO), "end": end.strftime(ISO), "sessions": int(len(curve))},
        "capital": {"initial": engine.initial_capital, "final": round(float(curve.iloc[-1]), 2)},
        "total_return": float(total_return),
        "cagr": float((1 + total_return) ** (1 / years) - 1),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0,
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
        "trades": int(len(trades)),
    }
    if len(trades):
        wins, losses = trades[trades["pnl"] > 0], trades[trades["pnl"] <= 0]
        report.update({
            "win_rate": float(len(wins) / len(trades)),
            "avg_win_pct": float(wins["pnl_pct"].mean()) if len(wins) else 0.0,
            "avg_loss_pct": float(losses["pnl_pct"].mean()) if len(losses) else 0.0,
            "profit_factor": float(wins["pnl"].sum() / -losses["pnl"].sum()) if losses["pnl"].sum() < 0 else None,
            "avg_bars_held": float(trades["bars_held"].mean()),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
            # calibration : la probabilité annoncée par XGBoost se vérifie-t-elle ?
            "avg_predicted_probability": float(trades["probability"].mean()),
            "target_hit_rate": float((trades["exit_reason"] == "OBJECTIF ATTEINT").mean()),
        })

    log_df = pd.DataFrame(decisions)
    counts = log_df["outcome"].value_counts().to_dict() if len(log_df) else {}
    committee = {"signals": int(len(log_df)), **{k.lower(): int(v) for k, v in counts.items()}}
    for outcome in ("APPROVED", "VETO"):  # valeur ajoutée du veto : que seraient devenus les signaux ?
        subset = log_df[(log_df["outcome"] == outcome) & log_df["hypothetical_ret"].notna()] if len(log_df) else log_df
        if len(subset):
            committee[f"{outcome.lower()}_hypothetical_avg_ret"] = float(subset["hypothetical_ret"].mean())
            committee[f"{outcome.lower()}_hypothetical_win_rate"] = float(subset["hypothetical_win"].mean())
    report["committee"] = committee

    benchmarks = {}
    spy = macro["spy"].loc[start:end].dropna() if "spy" in macro else pd.Series(dtype=float)
    if len(spy) > 1:
        benchmarks["spy_buy_and_hold"] = float(spy.iloc[-1] / spy.iloc[0] - 1)
    universe = [p["close"].loc[start:end] for p in prices.values()]
    universe = [s.iloc[-1] / s.iloc[0] - 1 for s in universe if len(s) > 1]
    if universe:
        benchmarks["universe_equal_weight"] = float(np.mean(universe))
    report["benchmarks"] = benchmarks
    return report


def print_report(report: dict[str, Any]) -> None:
    pct = lambda v: f"{v:+.2%}"  # noqa: E731
    line = "=" * 62
    print(f"\n{line}\n  BILAN DU BACKTEST  {report['period']['start']} -> {report['period']['end']}\n{line}")
    print(f"  Capital            : {report['capital']['initial']:,.0f} -> {report['capital']['final']:,.2f} USD")
    print(f"  Rendement total    : {pct(report['total_return'])}   (annualisé {pct(report['cagr'])})")
    print(f"  Ratio de Sharpe    : {report['sharpe']:.2f}")
    print(f"  Perte max (DD)     : {pct(report['max_drawdown'])}")
    print(f"  Trades clôturés    : {report['trades']}")
    if report["trades"]:
        pf = report["profit_factor"]
        print(f"  Taux de réussite   : {report['win_rate']:.1%}   (gain moyen {pct(report['avg_win_pct'])}, "
              f"perte moyenne {pct(report['avg_loss_pct'])})")
        print(f"  Profit factor      : {pf:.2f}" if pf is not None else "  Profit factor      : n/d (aucune perte)")
        print(f"  Calibration        : probabilité moyenne annoncée {report['avg_predicted_probability']:.1%} "
              f"vs objectif réellement atteint {report['target_hit_rate']:.1%}")
        print(f"  Sorties            : {report['exit_reasons']}")
    c = report["committee"]
    print(f"  Signaux XGBoost    : {c['signals']}  (autorisés {c.get('approved', 0)}, veto {c.get('veto', 0)}, "
          f"comité en erreur {c.get('error', 0)}, sans place {c.get('no_slot', 0)})")
    if "veto_hypothetical_avg_ret" in c:
        print(f"  Qualité du veto    : les signaux bloqués auraient rapporté {pct(c['veto_hypothetical_avg_ret'])} en moyenne"
              + (f" (autorisés : {pct(c['approved_hypothetical_avg_ret'])})" if "approved_hypothetical_avg_ret" in c else ""))
    for name, value in report["benchmarks"].items():
        print(f"  Référence {name:<24}: {pct(value)}")
    print(line)


# ---------------------------------------------------------------------------
# Boucle temporelle
# ---------------------------------------------------------------------------
def run_backtest(start: str, end: str, tickers: list[str], *, capital: float, use_llm: bool, use_memory: bool,
                 autopsy: bool, retrain_days: int, threshold: Optional[float] = None) -> dict[str, Any]:
    cfg = StrategyConfig.from_env(threshold=threshold)
    run_id = f"bt-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"Backtest {run_id} | {start} -> {end} | {len(tickers)} valeurs | seuil {cfg.threshold:.0%} | "
          f"objectif {cfg.tp_atr} ATR / stop {cfg.sl_atr} ATR / {cfg.horizon} séances | "
          f"comité IA : {'actif' if use_llm else 'désactivé (--no-llm)'}", flush=True)

    sync_all(tickers, end=end)
    if os.getenv("ENABLE_DARK_POOL", "false").lower() == "true":
        sync_dark_pool(tickers, os.getenv("HISTORY_START", "2016-01-01"), end)
    dataset, prices, features = build_dataset(tickers, cfg, end=end)
    macro = load_macro(end=end)
    days = [d for d in dataset.index.unique().sort_values() if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(days) < 2:
        raise SystemExit("Période de test vide : vérifiez les dates et la base de prix.")
    if use_memory:
        from data_pipeline.qdrant_setup import ensure_collections

        ensure_collections()

    engine = ExecutionEngine(cfg, capital, run_id, mode="backtest")
    decisions: list[dict[str, Any]] = []
    model, sessions_since_training, lessons = None, 0, 0
    by_day = dict(tuple(dataset.groupby(level=0)))

    for i, day in enumerate(days):
        rows = by_day[day]
        bars = {r["ticker"]: r for _, r in rows.iterrows()}

        closed_today = engine.on_open(day, bars)
        closed_today += engine.on_session(day, bars)
        equity = engine.mark(day, bars)

        # Walk-forward : ré-entraînement sur les seuls trades dont l'issue est connue ce soir
        if model is None or sessions_since_training >= retrain_days:
            train = training_rows(dataset, day)
            assert train["label_end"].max() <= day, "fuite temporelle dans l'échantillon d'entraînement"
            model, sessions_since_training = train_model(train, features), 0
            log.info("[%s] XGBoost ré-entraîné sur %d trades simulés (jusqu'au %s)", day.date(), len(train),
                     train["label_end"].max().date())
        sessions_since_training += 1

        if i < len(days) - 1:  # pas de nouveau signal le dernier jour : aucune ouverture pour l'exécuter
            approved, systemic = evaluate_signals(
                day, rows, model, features, cfg, prices=prices, macro=macro,
                held=set(engine.positions) | set(engine.pending), cooldown=engine.cooldown, equity=equity,
                cash=engine.buying_power(), slots=engine.free_slots(), mode="backtest", scope=run_id,
                use_llm=use_llm, use_memory=use_memory, decisions=decisions)
            for order in approved:
                engine.pending[order.plan.ticker] = order
            if systemic:
                engine.emergency_exit = True
                log.info("[%s] ALERTE SYSTÉMIQUE : positions coupées à la prochaine ouverture.", day.date())

        if use_llm and autopsy and closed_today:
            lessons += night_autopsy(closed_today, prices, macro, mode="backtest", scope=run_id, use_memory=use_memory)

    engine.liquidate(days[-1], "FIN DE SIMULATION")
    engine.equity_curve[-1] = (days[-1].strftime(ISO), engine.equity())

    # Contrôles d'intégrité de la boucle temporelle et de la comptabilité
    assert all(a < b for a, b in zip(days, days[1:])), "séances non chronologiques"
    assert not engine.positions, "positions résiduelles"
    pnl_total = sum(t["pnl"] for t in engine.closed)
    assert abs(engine.cash - (engine.initial_capital + pnl_total)) < 1e-6 * engine.initial_capital, "écart comptable"

    report = performance_report(engine, decisions, macro, prices)
    report.update({"run_id": run_id, "lessons_recorded": lessons, "llm": use_llm, "tickers": tickers,
                   "features": features, "strategy": asdict(cfg), "integrity_checks": "ok"})
    out_dir = REPORT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    pd.DataFrame(engine.equity_curve, columns=["date", "equity"]).to_csv(out_dir / "equity.csv", index=False)
    pd.DataFrame(decisions).to_csv(out_dir / "decisions.csv", index=False)
    if engine.closed:
        pd.DataFrame(engine.closed).drop(columns=["context"]).to_csv(out_dir / "trades.csv", index=False)
    print_report(report)
    print(f"  Leçons enregistrées dans le journal : {lessons}\n  Rapport : {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward du hedge fund IA.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--capital", type=float, default=float(os.getenv("INITIAL_CAPITAL", "100000")))
    parser.add_argument("--threshold", type=float, default=None, help="Seuil de probabilité (défaut : .env).")
    parser.add_argument("--retrain-days", type=int, default=21, help="Séances entre deux ré-entraînements.")
    parser.add_argument("--no-llm", action="store_true", help="Moteur quantitatif seul : tout signal est exécuté.")
    parser.add_argument("--no-memory", action="store_true", help="Sans Qdrant (ni dépêches, ni journal d'expérience).")
    parser.add_argument("--no-autopsy", action="store_true", help="Sans autopsie nocturne des pertes.")
    parser.add_argument("--quiet", action="store_true", help="N'affiche que le bilan final.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "sentence_transformers", "urllib3", "yfinance", "history"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    use_llm = not args.no_llm
    if use_llm:
        from llm_agents.graph import check_llm_services

        problems = check_llm_services()
        if problems:
            print("Le comité IA ne peut pas siéger :")
            print("\n".join(f"  - {p}" for p in problems))
            print("Corrigez la configuration (.env) ou lancez `python run_backtest.py --no-llm` "
                  "pour tester le moteur quantitatif seul.")
            sys.exit(2)

    run_backtest(args.start, args.end, [t.upper() for t in args.tickers], capital=args.capital, use_llm=use_llm,
                 use_memory=use_llm and not args.no_memory, autopsy=not args.no_autopsy,
                 retrain_days=args.retrain_days, threshold=args.threshold)


if __name__ == "__main__":
    main()
