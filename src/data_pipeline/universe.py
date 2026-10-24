"""Multi-market instrument universe: NSE India plus US large caps.

Adding a second market is not just a longer ticker list. Each market brings its own
currency, session, settlement, fee schedule and tick regime, and a system that silently
applies Indian STT to a US trade would produce confident nonsense. So a market is a
first-class object here, and anything market-specific is looked up rather than assumed.

What differs, and is therefore modelled:

- **Fees.** The NSE schedule in `EquityCosts` — STT, stamp duty, SEBI turnover fee, GST —
  has no US analogue. US retail equity trading is typically zero-commission with a small
  SEC/FINRA pass-through on sells. Using the Indian schedule for a US stock would overstate
  costs by roughly an order of magnitude.
- **Shorting.** The intraday-only rule is an NSE cash-market rule. It does not apply to a
  US margin account, so it is a property of the market, not a global constant.
- **Currency.** Position sizing in rupees against a dollar-denominated stock is wrong. Each
  market declares its currency; mixing them in one portfolio needs an FX rate and is
  deliberately **not** supported.
- **Session and calendar.** 09:15-15:30 IST versus 09:30-16:00 ET, different holidays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging

from src.paths import CONFIG_DIR, INSTRUMENTS_PATH, WATCHLIST_PATH

logger = logging.getLogger(__name__)

US_INSTRUMENTS_PATH = CONFIG_DIR / "instruments_us.json"


@dataclass(frozen=True)
class Market:
    code: str
    name: str
    currency: str
    symbol_suffix: str
    session_open: str
    session_close: str
    timezone: str
    intraday_short_only: bool
    default_tick: float
    index_symbol: str
    cost_profile: str

    def yahoo_symbol(self, ticker: str) -> str:
        return f"{ticker}{self.symbol_suffix}"


NSE = Market(
    code="NSE", name="National Stock Exchange of India", currency="INR",
    symbol_suffix=".NS", session_open="09:15", session_close="15:30",
    timezone="Asia/Kolkata", intraday_short_only=True, default_tick=0.05,
    index_symbol="^NSEI", cost_profile="nse_equity")

US = Market(
    code="US", name="US equities (NASDAQ/NYSE)", currency="USD",
    symbol_suffix="", session_open="09:30", session_close="16:00",
    timezone="America/New_York", intraday_short_only=False, default_tick=0.01,
    index_symbol="^GSPC", cost_profile="us_equity")

MARKETS = {market.code: market for market in (NSE, US)}

# Mega/large caps across sectors. Deliberately liquid names only: the risk model's
# participation and liquidity terms assume a bar's turnover is meaningful.
US_SEED = {
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corporation", "Technology"),
    "NVDA": ("NVIDIA Corporation", "Technology"),
    "GOOGL": ("Alphabet Inc.", "Communication Services"),
    "AMZN": ("Amazon.com, Inc.", "Consumer Cyclical"),
    "META": ("Meta Platforms, Inc.", "Communication Services"),
    "TSLA": ("Tesla, Inc.", "Consumer Cyclical"),
    "AVGO": ("Broadcom Inc.", "Technology"),
    "AMD": ("Advanced Micro Devices, Inc.", "Technology"),
    "INTC": ("Intel Corporation", "Technology"),
    "CRM": ("Salesforce, Inc.", "Technology"),
    "ORCL": ("Oracle Corporation", "Technology"),
    "ADBE": ("Adobe Inc.", "Technology"),
    "NFLX": ("Netflix, Inc.", "Communication Services"),
    "JPM": ("JPMorgan Chase & Co.", "Financial Services"),
    "BAC": ("Bank of America Corporation", "Financial Services"),
    "GS": ("The Goldman Sachs Group, Inc.", "Financial Services"),
    "V": ("Visa Inc.", "Financial Services"),
    "MA": ("Mastercard Incorporated", "Financial Services"),
    "BRK-B": ("Berkshire Hathaway Inc.", "Financial Services"),
    "UNH": ("UnitedHealth Group Incorporated", "Healthcare"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "LLY": ("Eli Lilly and Company", "Healthcare"),
    "PFE": ("Pfizer Inc.", "Healthcare"),
    "MRK": ("Merck & Co., Inc.", "Healthcare"),
    "XOM": ("Exxon Mobil Corporation", "Energy"),
    "CVX": ("Chevron Corporation", "Energy"),
    "WMT": ("Walmart Inc.", "Consumer Defensive"),
    "COST": ("Costco Wholesale Corporation", "Consumer Defensive"),
    "PG": ("The Procter & Gamble Company", "Consumer Defensive"),
    "KO": ("The Coca-Cola Company", "Consumer Defensive"),
    "PEP": ("PepsiCo, Inc.", "Consumer Defensive"),
    "HD": ("The Home Depot, Inc.", "Consumer Cyclical"),
    "MCD": ("McDonald's Corporation", "Consumer Cyclical"),
    "NKE": ("NIKE, Inc.", "Consumer Cyclical"),
    "DIS": ("The Walt Disney Company", "Communication Services"),
    "BA": ("The Boeing Company", "Industrials"),
    "CAT": ("Caterpillar Inc.", "Industrials"),
    "GE": ("GE Aerospace", "Industrials"),
    "UPS": ("United Parcel Service, Inc.", "Industrials"),
}

# Single-token aliases that are common English words or too generic to match safely.
# Same rule as the NSE alias builder: a bad match corrupts a reading, a miss only costs
# a zero penalty.
_UNSAFE_SINGLE_TOKEN = {"apple", "meta", "visa", "oracle", "target", "gap", "amazon",
                        "disney", "intel", "shell", "ge", "v", "ma", "key"}

_SUFFIXES = (", Inc.", " Inc.", ", Incorporated", " Incorporated", " Corporation",
             " Corp.", " Company", " & Co.", " Co.", " plc", " Ltd.", " Limited")


def _us_aliases(ticker: str, name: str) -> list:
    aliases = [name]
    trimmed = name
    for suffix in _SUFFIXES:
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)].strip().rstrip(",")
            break
    if trimmed and trimmed != name:
        aliases.append(trimmed)
    # Keep multi-token forms, plus single tokens that are not ordinary words.
    safe = []
    for alias in aliases:
        if len(alias.split()) >= 2 or alias.lower() not in _UNSAFE_SINGLE_TOKEN:
            if alias not in safe:
                safe.append(alias)
    return safe


def build_us_instruments() -> dict:
    instruments = {}
    for ticker, (name, sector) in US_SEED.items():
        instruments[ticker] = {
            "company_name": name, "yahoo_symbol": ticker, "sector": sector,
            "aliases": _us_aliases(ticker, name), "tick_size": US.default_tick,
            "lot_size": 1, "feed_available": True, "market": "US", "currency": "USD",
        }
    return instruments


def write_us_instruments(path=None) -> dict:
    path = path or US_INSTRUMENTS_PATH
    instruments = build_us_instruments()
    payload = {
        "schema_version": 1, "market": "US", "currency": "USD",
        "default_tick_size": US.default_tick,
        "alias_rule": "Generated aliases keep >=2 tokens unless the single token is "
                      "distinctive; ordinary English words (Apple, Meta, Visa) are excluded "
                      "because a false match corrupts a reading while a miss costs nothing.",
        "instruments": instruments,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return instruments


def load_market_instruments(market_code: str) -> dict:
    """Instruments for one market, each tagged with its market and currency."""
    if market_code == "NSE":
        if not INSTRUMENTS_PATH.exists():
            tickers = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8")).get("watchlist", [])
            return {t: {"company_name": t, "aliases": [t], "market": "NSE",
                        "currency": "INR", "tick_size": NSE.default_tick,
                        "feed_available": True} for t in tickers}
        payload = json.loads(INSTRUMENTS_PATH.read_text(encoding="utf-8"))
        return {t: {**meta, "market": "NSE", "currency": "INR"}
                for t, meta in payload.get("instruments", {}).items()}
    if market_code == "US":
        if US_INSTRUMENTS_PATH.exists():
            payload = json.loads(US_INSTRUMENTS_PATH.read_text(encoding="utf-8"))
            return {t: {**meta, "market": "US", "currency": "USD"}
                    for t, meta in payload.get("instruments", {}).items()}
        return build_us_instruments()
    raise ValueError(f"Unknown market: {market_code}")


def load_universe(markets=("NSE",)) -> dict:
    """Instruments across the requested markets, keyed `MARKET:TICKER`.

    The key is qualified because ticker symbols collide across markets and a bare symbol
    would silently resolve to whichever market loaded last.
    """
    universe = {}
    for code in markets:
        for ticker, meta in load_market_instruments(code).items():
            universe[f"{code}:{ticker}"] = {**meta, "ticker": ticker, "market": code}
    return universe


def split_key(key: str) -> tuple:
    """'NSE:RELIANCE' -> ('NSE', 'RELIANCE'); a bare symbol defaults to NSE."""
    if ":" in key:
        code, ticker = key.split(":", 1)
        if code in MARKETS:
            return code, ticker
    return "NSE", key


def market_for(key: str) -> Market:
    return MARKETS[split_key(key)[0]]


def yahoo_symbol_for(key: str, instruments: dict = None) -> str:
    code, ticker = split_key(key)
    meta = (instruments or {}).get(key) or (instruments or {}).get(ticker) or {}
    explicit = meta.get("yahoo_symbol")
    if explicit:
        return explicit
    return MARKETS[code].yahoo_symbol(ticker)


def costs_for(market_code: str, intraday: bool = False, slippage_bps: float = 5.0):
    """The right fee schedule for the market. Never apply NSE charges to a US trade."""
    from src.quant_engine.research import EquityCosts

    if market_code == "NSE":
        return EquityCosts(intraday=intraday, slippage_bps=slippage_bps)
    if market_code == "US":
        # Zero-commission retail brokerage. What remains is the SEC Section 31 fee plus
        # the FINRA Trading Activity Fee, both charged on sells only and both small —
        # roughly 0.0028% of sell turnover combined. No stamp duty, no GST, no DP charge.
        # Verify against your own broker; these are approximate published rates.
        return EquityCosts(
            intraday=intraday, brokerage_rate=0.0, brokerage_cap=0.0,
            exchange_rate=0.0, sebi_rate=0.0, gst_rate=0.0, delivery_dp=0.0,
            stt_intraday_sell=0.0000278, stt_delivery=0.0000278,
            stt_delivery_sell_only=True,
            stamp_intraday_buy=0.0, stamp_delivery_buy=0.0,
            slippage_bps=slippage_bps)
    raise ValueError(f"Unknown market: {market_code}")
