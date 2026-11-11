"""Point d'entrée de l'exécution réelle (Broker API) — paper-trading par défaut.

Cycle quotidien, à lancer après la clôture de Wall Street :
    1. mise à jour des prix, des options, des dépêches RSS et des dépôts SEC ;
    2. gestion des positions ouvertes (stop-loss, objectifs, durée max, arrêt d'urgence) ;
    3. scan XGBoost  ->  dossier (RAG)  ->  débat Bull/Bear  ->  veto du CRO  ->  Kelly ;
    4. envoi des ordres au broker pour la prochaine ouverture ;
    5. autopsie nocturne des trades perdants (RAG d'expérience).

Brokers (variable BROKER du .env) :
    paper  : simulateur local, même moteur que le backtest (aucun compte requis) ;
    alpaca : API Alpaca. Seule l'URL de paper-trading est acceptée, sauf ALLOW_LIVE_TRADING=true.

Usage :
    python run_live.py                 # un cycle
    python run_live.py --dry-run       # analyse sans rien enregistrer ni envoyer
    python run_live.py --loop --at 16:30   # un cycle par jour ouvré (heure de New York)
    python run_live.py --autopsy       # autopsie seule ; --status : état du portefeuille
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from data_pipeline.fetch_history import (
    TICKERS, db_connect, load_macro, snapshot_options_flow, sync_all, sync_dark_pool,
)
from quant_engine.xgboost_model import (
    StrategyConfig, TradePlan, build_dataset, load_model, save_model, train_model, training_rows,
)
from run_backtest import ISO, ExecutionEngine, _journal_trade, evaluate_signals, night_autopsy

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

log = logging.getLogger("live")

STATE_PATH = ROOT / "data" / "live_state.json"
MARKET_TZ = "America/New_York"
RETRAIN_AFTER_DAYS = int(os.getenv("LIVE_RETRAIN_DAYS", "7"))


# ---------------------------------------------------------------------------
# Données et modèle
# ---------------------------------------------------------------------------
def refresh_data(tickers: list[str], with_news: bool) -> None:
    sync_all(tickers)  # end=None : jusqu'à la dernière séance
    for ticker in tickers:
        snapshot_options_flow(ticker)
    if os.getenv("ENABLE_DARK_POOL", "false").lower() == "true":
        sync_dark_pool(tickers, os.getenv("HISTORY_START", "2016-01-01"))
    if with_news:
        from data_pipeline.fetch_news import ingest_daily

        report = ingest_daily(tickers)
        log.info("Mémoire : %d dépêches RSS, %d dépôts SEC ingérés.", report["rss"], report["sec_filings"])


def completed_sessions(dataset: pd.DataFrame, allow_intraday: bool) -> pd.DataFrame:
    """Écarte la bougie du jour si la séance n'est pas terminée (elle serait partielle)."""
    now = pd.Timestamp.now(tz=MARKET_TZ)
    last = dataset.index.max()
    if not allow_intraday and last.date() == now.date() and now.hour < 16:
        log.warning("Séance du %s encore ouverte : le cycle s'appuie sur la clôture précédente.", last.date())
        return dataset[dataset.index < last]
    return dataset


def ensure_model(dataset: pd.DataFrame, features: list[str], day: pd.Timestamp):
    """Charge le modèle sauvegardé ; ré-entraîne (hyperparamètres figés par Optuna) s'il est périmé."""
    model, meta = load_model()
    fresh = (model is not None and meta.get("features") == features
             and (day - pd.Timestamp(meta["trained_until"])).days <= RETRAIN_AFTER_DAYS)
    if fresh:
        return model
    train = training_rows(dataset, day)
    model = train_model(train, features)
    save_model(model, features, {"trained_until": day.strftime(ISO), "rows": len(train),
                                 "trained_at": datetime.now().isoformat(timespec="seconds")})
    log.info("XGBoost ré-entraîné sur %d trades simulés (jusqu'au %s).", len(train), day.date())
    return model


# ---------------------------------------------------------------------------
# Broker Alpaca (API REST v2)
# ---------------------------------------------------------------------------
class AlpacaBroker:
    def __init__(self) -> None:
        self.base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        key, secret = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            raise SystemExit("BROKER=alpaca : renseignez ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env.")
        if "paper-api" not in self.base and os.getenv("ALLOW_LIVE_TRADING", "false").lower() != "true":
            raise SystemExit("URL Alpaca hors paper-trading refusée (ALLOW_LIVE_TRADING=true pour passer en réel).")
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self.session.request(method, f"{self.base}{path}", timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def account(self) -> tuple[float, float]:
        acc = self._call("GET", "/v2/account")
        return float(acc["equity"]), float(acc["cash"])

    def positions(self) -> dict[str, dict[str, float]]:
        return {p["symbol"]: {"shares": float(p["qty"]), "entry_price": float(p["avg_entry_price"])}
                for p in self._call("GET", "/v2/positions")}

    def submit_bracket(self, plan: TradePlan) -> str:
        """Achat au marché à la prochaine ouverture, avec stop-loss et objectif attachés."""
        stop, target = plan.levels(plan.reference_price)
        order = self._call("POST", "/v2/orders", json={
            "symbol": plan.ticker, "qty": str(plan.shares), "side": "buy", "type": "market",
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": f"{target:.2f}"}, "stop_loss": {"stop_price": f"{stop:.2f}"},
        })
        return order["id"]

    def order(self, order_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v2/orders/{order_id}")

    def last_sell(self, ticker: str) -> Optional[dict[str, Any]]:
        orders = self._call("GET", "/v2/orders", params={"status": "closed", "symbols": ticker,
                                                         "limit": 50, "direction": "desc"})
        return next((o for o in orders if o["side"] == "sell" and o.get("filled_avg_price")), None)

    def close(self, ticker: str) -> None:
        for order in self._call("GET", "/v2/orders", params={"status": "open", "symbols": ticker}):
            self._call("DELETE", f"/v2/orders/{order['id']}")  # libère les titres bloqués par le bracket
        time.sleep(1.0)  # l'annulation est asynchrone côté broker
        self._call("DELETE", f"/v2/positions/{ticker}")

    def close_all(self) -> None:
        self._call("DELETE", "/v2/positions", params={"cancel_orders": "true"})


# ---------------------------------------------------------------------------
# État persistant
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))


def committee_ready() -> bool:
    from llm_agents.graph import check_llm_services

    problems = check_llm_services()
    for problem in problems:
        log.warning("Comité IA : %s", problem)
    if problems:
        log.warning("Aucun nouveau trade tant que le comité ne peut pas siéger (les positions restent surveillées).")
    return not problems


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------
def cycle_paper(tickers: list[str], cfg: StrategyConfig, dry_run: bool, allow_intraday: bool) -> None:
    dataset, prices, features = build_dataset(tickers, cfg)
    dataset = completed_sessions(dataset, allow_intraday)
    macro = load_macro()
    today = dataset.index.max()
    macro = macro.loc[:today]

    state = load_state().get("paper")
    engine = (ExecutionEngine.from_state(state["engine"], cfg, "live") if state
              else ExecutionEngine(cfg, float(os.getenv("INITIAL_CAPITAL", "100000")), "live-paper", "live"))
    last_done = pd.Timestamp(state["last_day"]) if state else None
    if last_done is not None and last_done >= today:
        print(f"Séance du {today.date()} déjà traitée. Capital : {engine.equity():,.2f} USD.")
        return
    sessions = dataset.index.unique().sort_values()
    todo = [today] if last_done is None else [d for d in sessions if d > last_done]

    by_day = dict(tuple(dataset.groupby(level=0)))
    closed: list[dict[str, Any]] = []
    for day in todo:  # rattrape aussi les séances manquées
        bars = {r["ticker"]: r for _, r in by_day[day].iterrows()}
        closed += engine.on_open(day, bars) + engine.on_session(day, bars)
        equity = engine.mark(day, bars)

    decisions: list[dict[str, Any]] = []
    ready = committee_ready()
    if ready:
        model = ensure_model(dataset, features, today)
        approved, systemic = evaluate_signals(
            today, by_day[today], model, features, cfg, prices=prices, macro=macro,
            held=set(engine.positions) | set(engine.pending), cooldown=engine.cooldown, equity=equity,
            cash=engine.buying_power(), slots=engine.free_slots(), mode="live", scope=None,
            use_llm=True, use_memory=True, decisions=decisions)
        for order in approved:
            engine.pending[order.plan.ticker] = order
        engine.emergency_exit = systemic

    print(f"\nCycle du {today.date()} — capital {equity:,.2f} USD, liquidités {engine.cash:,.2f} USD")
    print(f"  Positions : {', '.join(f'{t} x{p.shares}' for t, p in engine.positions.items()) or 'aucune'}")
    print(f"  Trades clôturés : {len(closed)} | signaux XGBoost : {len(decisions)} | "
          f"ordres pour la prochaine ouverture : {', '.join(engine.pending) or 'aucun'}")
    if engine.emergency_exit:
        print("  ALERTE SYSTÉMIQUE du CRO : toutes les positions seront coupées à la prochaine ouverture.")
    if dry_run:
        print("  (--dry-run : rien n'a été enregistré)")
        return
    save_state({**load_state(), "paper": {"last_day": today.strftime(ISO), "engine": engine.to_state()}})
    if ready and closed:
        night_autopsy(closed, prices, macro, mode="live", scope="live", use_memory=True)


def cycle_alpaca(tickers: list[str], cfg: StrategyConfig, dry_run: bool, allow_intraday: bool) -> None:
    broker = AlpacaBroker()
    dataset, prices, features = build_dataset(tickers, cfg)
    dataset = completed_sessions(dataset, allow_intraday)
    macro = load_macro()
    today = dataset.index.max()
    macro = macro.loc[:today]
    today_iso = today.strftime(ISO)

    state = load_state()
    book = state.get("alpaca", {"trades": {}, "cooldown": {}})
    equity, cash = broker.account()
    positions = broker.positions()
    closed: list[dict[str, Any]] = []

    # 1. Rapprochement avec le broker : ordres exécutés, positions fermées par le bracket
    for ticker, trade in list(book["trades"].items()):
        if ticker in positions:
            if trade["status"] == "submitted":
                trade.update(status="open", entry_date=today_iso, entry_price=positions[ticker]["entry_price"],
                             shares=positions[ticker]["shares"])
            sessions_held = int((prices[ticker].loc[trade["entry_date"]:today].shape[0]))
            if sessions_held >= cfg.horizon and not dry_run:  # 2. durée max
                broker.close(ticker)
                log.info("%s : durée max atteinte, position soldée.", ticker)
            continue
        entry_order = broker.order(trade["order_id"])
        status = entry_order["status"]
        if trade["status"] == "submitted" and status in ("new", "accepted", "pending_new", "partially_filled"):
            continue  # pas encore exécuté
        del book["trades"][ticker]
        if trade["status"] == "submitted" and status == "filled":  # entré puis ressorti entre deux cycles
            trade.update(status="open", entry_price=float(entry_order["filled_avg_price"]),
                         shares=float(entry_order["filled_qty"]))
        sell = broker.last_sell(ticker) if trade["status"] == "open" else None
        if sell is None:
            log.info("%s : ordre %s sans exécution (%s).", ticker, trade["order_id"], status)
            continue
        exit_price = float(sell["filled_avg_price"])
        reason = {"limit": "OBJECTIF ATTEINT", "stop": "STOP-LOSS"}.get(sell["type"], "SORTIE AU MARCHÉ")
        record = {**trade, "run_id": "live-alpaca", "mode": "live", "ticker": ticker, "exit_date": today_iso,
                  "exit_price": exit_price, "exit_reason": reason,
                  "pnl": (exit_price - trade["entry_price"]) * trade["shares"],
                  "pnl_pct": exit_price / trade["entry_price"] - 1}
        if not dry_run:
            record["id"] = _journal_trade(record)
            closed.append(record)
        log.info("%s : sortie %s à %.2f (%+.2f%%).", ticker, reason, exit_price, record["pnl_pct"] * 100)

    # 3. Nouveaux signaux
    decisions: list[dict[str, Any]] = []
    ready = committee_ready()
    if ready:
        model = ensure_model(dataset, features, today)
        rows = dataset.loc[[today]]
        held = set(positions) | set(book["trades"])
        approved, systemic = evaluate_signals(
            today, rows, model, features, cfg, prices=prices, macro=macro, held=held, cooldown=book["cooldown"],
            equity=equity, cash=cash, slots=cfg.max_open_positions - len(held), mode="live", scope=None,
            use_llm=True, use_memory=True, decisions=decisions)
        if systemic and not dry_run:
            broker.close_all()
            book["trades"] = {t: v for t, v in book["trades"].items() if v["status"] == "open"}
            log.warning("ALERTE SYSTÉMIQUE du CRO : fermeture de toutes les positions demandée au broker.")
        for order in approved if not systemic else []:
            plan = order.plan
            stop, target = plan.levels(plan.reference_price)
            order_id = "dry-run" if dry_run else broker.submit_bracket(plan)
            book["trades"][plan.ticker] = {
                "status": "submitted", "order_id": order_id, "signal_date": today_iso, "entry_date": today_iso,
                "entry_price": plan.reference_price, "shares": plan.shares, "stop_loss": stop, "take_profit": target,
                "probability": plan.probability, "kelly_fraction": plan.kelly_full,
                "allocation_pct": plan.allocation_pct, "context": order.context,
            }
            print(f"  Ordre bracket envoyé : {plan.ticker} x{plan.shares} | stop {stop:.2f} | objectif {target:.2f}")

    print(f"\nCycle Alpaca du {today.date()} — capital {equity:,.2f} USD, liquidités {cash:,.2f} USD, "
          f"{len(positions)} position(s), {len(decisions)} signal(aux), {len(closed)} trade(s) clôturé(s).")
    if dry_run:
        print("  (--dry-run : aucun ordre envoyé, rien n'a été enregistré)")
        return
    save_state({**state, "alpaca": book})
    if ready and closed:
        night_autopsy(closed, prices, macro, mode="live", scope="live", use_memory=True)


def autopsy_backlog(tickers: list[str], cfg: StrategyConfig) -> None:
    """Rattrape les autopsies des trades live perdants jamais analysés."""
    with db_connect() as con:
        rows = con.execute("SELECT * FROM trades WHERE mode = 'live' AND pnl < 0 AND autopsy_done = 0").fetchall()
    if not rows:
        print("Aucun trade perdant en attente d'autopsie.")
        return
    if not committee_ready():
        return
    trades = [{**dict(r), "context": json.loads(r["context_json"] or "{}")} for r in rows]
    _, prices, _ = build_dataset(sorted({t["ticker"] for t in trades} | set(tickers)), cfg)
    done = night_autopsy(trades, prices, load_macro(), mode="live", scope="live", use_memory=True)
    print(f"Autopsies réalisées : {done}/{len(trades)}")


def show_status() -> None:
    state = load_state()
    if "paper" in state:
        eng = state["paper"]["engine"]
        value = eng["cash"] + sum(p["shares"] * eng["last_price"].get(t, p["entry_price"])
                                  for t, p in eng["positions"].items())
        print(f"[paper] dernière séance traitée : {state['paper']['last_day']} | capital {value:,.2f} USD "
              f"(départ {eng['initial_capital']:,.0f}) | liquidités {eng['cash']:,.2f}")
        for ticker, pos in eng["positions"].items():
            print(f"  {ticker:<6} x{pos['shares']} entrée {pos['entry_price']:.2f} le {pos['entry_date']} | "
                  f"stop {pos['stop_loss']:.2f} | objectif {pos['take_profit']:.2f}")
        print(f"  Ordres en attente : {', '.join(eng['pending']) or 'aucun'}")
    if "alpaca" in state:
        print(f"[alpaca] trades suivis : {', '.join(state['alpaca']['trades']) or 'aucun'}")
    if not state:
        print("Aucun cycle live exécuté pour l'instant.")
    with db_connect() as con:
        row = con.execute("SELECT COUNT(*), COALESCE(SUM(pnl), 0), COALESCE(SUM(pnl > 0), 0) "
                          "FROM trades WHERE mode = 'live'").fetchone()
    print(f"Journal live : {row[0]} trade(s) clôturé(s), P&L {row[1]:+,.2f} USD, {row[2]} gagnant(s).")


def run_cycle(args: argparse.Namespace) -> None:
    cfg = StrategyConfig.from_env()
    tickers = [t.upper() for t in args.tickers]
    refresh_data(tickers, with_news=not args.no_news)
    broker = os.getenv("BROKER", "paper").lower()
    cycle = cycle_alpaca if broker == "alpaca" else cycle_paper
    cycle(tickers, cfg, args.dry_run, args.allow_intraday)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exécution live (paper-trading) du hedge fund IA.")
    parser.add_argument("--tickers", nargs="*", default=TICKERS)
    parser.add_argument("--dry-run", action="store_true", help="Analyse sans enregistrer ni envoyer d'ordre.")
    parser.add_argument("--no-news", action="store_true", help="Ne pas rafraîchir les flux RSS / SEC.")
    parser.add_argument("--allow-intraday", action="store_true", help="Utiliser la bougie du jour même avant la clôture.")
    parser.add_argument("--loop", action="store_true", help="Un cycle par jour ouvré, indéfiniment.")
    parser.add_argument("--at", default="16:30", help="Heure du cycle en mode --loop (heure de New York).")
    parser.add_argument("--autopsy", action="store_true", help="Lancer uniquement l'autopsie des pertes en attente.")
    parser.add_argument("--status", action="store_true", help="Afficher l'état du portefeuille live.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "sentence_transformers", "urllib3", "yfinance", "history"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.status:
        return show_status()
    if args.autopsy:
        return autopsy_backlog([t.upper() for t in args.tickers], StrategyConfig.from_env())
    if not args.loop:
        return run_cycle(args)

    hour, minute = (int(x) for x in args.at.split(":"))
    print(f"Mode boucle : un cycle chaque jour ouvré à {args.at} (New York). Ctrl+C pour arrêter.")
    while True:
        now = pd.Timestamp.now(tz=MARKET_TZ)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        while target <= now or target.weekday() >= 5:
            target += pd.Timedelta(days=1)
        log.info("Prochain cycle : %s", target.strftime("%Y-%m-%d %H:%M %Z"))
        time.sleep(max((target - pd.Timestamp.now(tz=MARKET_TZ)).total_seconds(), 1))
        try:
            run_cycle(args)
        except Exception:
            log.exception("Cycle interrompu par une erreur ; nouvel essai au prochain créneau.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
