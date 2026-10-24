"""Append-only paper-trading book. No broker, no real orders.

Deliberately NOT an order-management system. One append-only table holds every event;
positions, cash and P&L are derived by query. That removes the entire class of bugs where
a positions row disagrees with the fills that produced it, needs no migrations, and means
a Streamlit rerun cannot corrupt state: writes are idempotent on a caller-supplied
event_id, so replaying the same action is a no-op.

Barrier resolution is delegated to research.resolve_exit, the same function the historical
replay uses, so the paper book and the backtest can never drift apart.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3

import pandas as pd

from src.paths import DB_PATH
from src.quant_engine.research import EquityCosts, resolve_exit

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_events (
    event_id         TEXT PRIMARY KEY,
    ts_ist           TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    interval         TEXT NOT NULL,
    direction        TEXT NOT NULL,
    event_type       TEXT NOT NULL CHECK (event_type IN ('SIGNAL', 'ENTRY', 'EXIT')),
    price            REAL,
    quantity         INTEGER,
    fees             REAL,
    product          TEXT,
    reason           TEXT,
    grade            TEXT,
    conviction       REAL,
    risk_score       REAL,
    strategy_version TEXT,
    report_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_events_key
    ON paper_events (ticker, interval, direction, ts_ist);
CREATE INDEX IF NOT EXISTS idx_paper_events_type ON paper_events (event_type);
"""

OUTCOME_EVENT = "target touched before stop"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval. Correct at small n, unlike the normal approximation."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def make_event_id(*parts) -> str:
    """Stable idempotency key. The same logical action always hashes the same."""
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperFill:
    ticker: str
    interval: str
    direction: str
    quantity: int
    entry_price: float
    stop: float
    target: float
    product: str = "DELIVERY"


class PaperBook:
    """Append-only event log plus derived views."""

    def __init__(self, db_path=None, starting_capital: float = 200000.0):
        self.db_path = str(db_path or DB_PATH)
        self.starting_capital = float(starting_capital)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---------------------------------------------------------------- writes

    def _append(self, event_id: str, **fields) -> bool:
        """True if written, False if this event_id already existed (idempotent)."""
        columns = ["event_id"] + list(fields)
        placeholders = ", ".join("?" for _ in columns)
        cursor = self.conn.execute(
            f"INSERT OR IGNORE INTO paper_events ({', '.join(columns)}) VALUES ({placeholders})",
            [event_id] + [fields[k] for k in fields])
        self.conn.commit()
        return cursor.rowcount > 0

    def record_signal(self, signal) -> str:
        """Log every generated signal, taken or not, so hit rate stays measurable."""
        payload = signal.to_dict()
        event_id = make_event_id("SIGNAL", signal.ticker, signal.interval, signal.action,
                                 signal.as_of_bar_close)
        self._append(
            event_id,
            ts_ist=signal.as_of_bar_close or pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
            ticker=signal.ticker, interval=signal.interval,
            direction=signal.direction, event_type="SIGNAL",
            grade=(signal.setup or {}).get("letter"),
            conviction=float(signal.conviction),
            risk_score=(signal.risk or {}).get("score"),
            reason=signal.status, product=(signal.plan or {}).get("product", "DELIVERY"),
            strategy_version=signal.version, report_json=json.dumps(payload, allow_nan=False))
        return event_id

    def open_from_signal(self, signal, costs: EquityCosts = None) -> str:
        """Record an ENTRY at the signal's own worst modelled fill.

        Rejects a SHORT whose product is not INTRADAY: NSE cash delivery cannot be sold
        short, and this is the third of three enforcement layers.
        """
        if not signal.plan:
            raise ValueError("Signal carries no plan; nothing to enter")
        product = signal.plan.get("product", "DELIVERY")
        if signal.direction == "SHORT" and product != "INTRADAY":
            raise ValueError("Short positions must be INTRADAY; NSE cash delivery cannot be shorted")
        costs = costs or EquityCosts(intraday=(product == "INTRADAY"))
        price = float(signal.plan["worst_modeled_entry_fill"])
        quantity = int(signal.plan["quantity"])
        side = "BUY" if signal.direction == "LONG" else "SELL"
        event_id = make_event_id("ENTRY", signal.ticker, signal.interval, signal.direction,
                                 signal.as_of_bar_close)
        self._append(
            event_id,
            ts_ist=signal.as_of_bar_close or pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
            ticker=signal.ticker, interval=signal.interval, direction=signal.direction,
            event_type="ENTRY", price=price, quantity=quantity,
            fees=float(costs.fee(price, quantity, side)), product=product,
            grade=(signal.setup or {}).get("letter"), conviction=float(signal.conviction),
            risk_score=(signal.risk or {}).get("score"), reason="signal_entry",
            strategy_version=signal.version,
            report_json=json.dumps({"stop": signal.plan["stop"],
                                    "target": signal.plan["take_profit"]}, allow_nan=False))
        return event_id

    def record_exit(self, ticker: str, interval: str, direction: str, price: float,
                    quantity: int, reason: str, ts_ist: str, product: str = "DELIVERY",
                    costs: EquityCosts = None) -> str:
        costs = costs or EquityCosts(intraday=(product == "INTRADAY"))
        side = "SELL" if direction == "LONG" else "BUY"
        event_id = make_event_id("EXIT", ticker, interval, direction, ts_ist, reason)
        self._append(event_id, ts_ist=ts_ist, ticker=ticker, interval=interval,
                     direction=direction, event_type="EXIT", price=float(price),
                     quantity=int(quantity), fees=float(costs.fee(price, quantity, side)),
                     product=product, reason=reason, strategy_version="paper-exit-v1")
        return event_id

    # ---------------------------------------------------------------- reads

    def events(self, event_type: str = None) -> pd.DataFrame:
        query = "SELECT * FROM paper_events"
        params = ()
        if event_type:
            query += " WHERE event_type = ?"
            params = (event_type,)
        query += " ORDER BY ts_ist, event_id"
        return pd.read_sql(query, self.conn, params=params)

    def _paired(self) -> tuple:
        """Walk the log per instrument, pairing each ENTRY with its next EXIT.

        Everything else is derived from this, so cash and positions always reconcile with
        the events on disk, including after a restart.
        """
        frame = self.events()
        fills = frame[frame.event_type.isin(["ENTRY", "EXIT"])]
        closed, open_rows = [], []
        for key, group in fills.groupby(["ticker", "interval", "direction"], sort=False):
            pending = None
            for row in group.sort_values(["ts_ist", "event_id"]).itertuples(index=False):
                if row.event_type == "ENTRY":
                    if pending is not None:
                        open_rows.append(pending)
                    pending = row
                elif pending is not None:
                    sign = 1.0 if row.direction == "LONG" else -1.0
                    gross = sign * (row.price - pending.price) * pending.quantity
                    fees = (pending.fees or 0.0) + (row.fees or 0.0)
                    closed.append({
                        "ticker": key[0], "interval": key[1], "direction": key[2],
                        "entry_ts": pending.ts_ist, "exit_ts": row.ts_ist,
                        "entry": pending.price, "exit": row.price,
                        "quantity": pending.quantity, "grade": pending.grade,
                        "conviction": pending.conviction, "risk_score": pending.risk_score,
                        "reason": row.reason, "fees": fees, "net_pnl": gross - fees,
                        "product": pending.product,
                    })
                    pending = None
            if pending is not None:
                open_rows.append(pending)
        open_frame = pd.DataFrame([{
            "ticker": r.ticker, "interval": r.interval, "direction": r.direction,
            "entry_ts": r.ts_ist, "entry": r.price, "quantity": r.quantity,
            "grade": r.grade, "conviction": r.conviction, "product": r.product,
            "fees": r.fees,
            "stop": (json.loads(r.report_json).get("stop") if r.report_json else None),
            "target": (json.loads(r.report_json).get("target") if r.report_json else None),
        } for r in open_rows])
        return pd.DataFrame(closed), open_frame

    def closed_trades(self) -> pd.DataFrame:
        return self._paired()[0]

    def open_positions(self) -> pd.DataFrame:
        return self._paired()[1]

    def cash(self) -> float:
        """Starting capital plus realised P&L. Recomputed from the log every call."""
        closed = self.closed_trades()
        return self.starting_capital + (float(closed.net_pnl.sum()) if not closed.empty else 0.0)

    def performance_summary(self) -> dict:
        closed = self.closed_trades()
        if closed.empty:
            return {"trades": 0, "starting_capital": self.starting_capital,
                    "cash": self.starting_capital, "note": "No closed paper trades yet."}
        pnl = closed.net_pnl
        gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        equity = self.starting_capital + pnl.cumsum()
        peak = equity.cummax().clip(lower=self.starting_capital)
        return {
            "trades": int(len(closed)),
            "starting_capital": self.starting_capital,
            "cash": float(self.starting_capital + pnl.sum()),
            "net_pnl": float(pnl.sum()),
            "net_return_pct": float(pnl.sum() / self.starting_capital * 100),
            "win_rate_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "expectancy_rupees": float(pnl.mean()),
            "total_fees": float(closed.fees.sum()),
            "max_drawdown_pct": float(((peak - equity) / peak).max() * 100),
            "by_grade": {str(g): int(n) for g, n in closed.grade.value_counts().items()},
        }

    def equity_curve(self) -> pd.DataFrame:
        closed = self.closed_trades()
        if closed.empty:
            return pd.DataFrame(columns=["exit_ts", "equity"])
        ordered = closed.sort_values("exit_ts")
        return pd.DataFrame({"exit_ts": ordered.exit_ts,
                             "equity": self.starting_capital + ordered.net_pnl.cumsum()})

    def hit_rate(self, direction: str, interval: str, grade: str,
                 minimum: int = 30) -> dict:
        """The only measured number in this system: this book's own realised hit rate.

        Returns available=False with the shortfall until the bucket has `minimum` closed
        trades, so the UI can say "n of 30 needed" instead of showing a made-up number.
        """
        closed = self.closed_trades()
        bucket = {"event": OUTCOME_EVENT, "direction": direction, "interval": interval,
                  "grade": grade, "minimum_samples": minimum}
        if closed.empty:
            return {**bucket, "available": False, "n": 0,
                    "reason": f"0 of {minimum} closed trades in this bucket"}
        matched = closed[(closed.direction == direction) & (closed.interval == interval)
                         & (closed.grade == grade)]
        total = int(len(matched))
        if total < minimum:
            return {**bucket, "available": False, "n": total,
                    "reason": f"{total} of {minimum} closed trades in this bucket"}
        wins = int((matched.reason == "target").sum())
        interval_bounds = wilson_interval(wins, total)
        return {**bucket, "available": True, "n": total, "wins": wins,
                "p": wins / total, "ci": [round(interval_bounds[0], 4),
                                          round(interval_bounds[1], 4)],
                "measured_on": str(matched.exit_ts.max()),
                "note": "Realised frequency in this paper book with a Wilson 95% interval. "
                        "Hypothetical fills, not broker executions."}

    def confidence_lookup(self, minimum: int = 30):
        """Adapter for composite.generate_signal(confidence_lookup=...)."""
        def lookup(direction: str, interval: str, grade: str) -> dict:
            return self.hit_rate(direction, interval, grade, minimum=minimum)
        return lookup

    # ------------------------------------------------------- position marking

    def evaluate_open_positions(self, candles: dict, costs: EquityCosts = None) -> list:
        """Walk each open position's bars after entry and close it at the first barrier.

        Uses research.resolve_exit, so stop-before-target and gap-stop-at-the-worse-open
        behave exactly as they do in the historical replay.
        """
        closures = []
        for position in self.open_positions().itertuples(index=False):
            frame = candles.get(position.ticker)
            if frame is None or frame.empty:
                continue
            if position.stop is None or position.target is None:
                continue
            bars = frame.copy()
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
            entry_ts = pd.Timestamp(position.entry_ts)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("Asia/Kolkata")
            after = bars[bars.timestamp > entry_ts.tz_convert("UTC")]
            for row in after.itertuples(index=False):
                price, reason = resolve_exit(row, float(position.stop),
                                             float(position.target), position.direction)
                if price is None:
                    continue
                product = position.product or "DELIVERY"
                self.record_exit(position.ticker, position.interval, position.direction,
                                 price=price, quantity=int(position.quantity), reason=reason,
                                 ts_ist=pd.Timestamp(row.timestamp).tz_convert(
                                     "Asia/Kolkata").isoformat(),
                                 product=product,
                                 costs=costs or EquityCosts(intraday=(product == "INTRADAY")))
                closures.append({"ticker": position.ticker, "reason": reason, "price": price})
                break
        return closures
