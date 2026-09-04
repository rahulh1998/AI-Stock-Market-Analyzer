import sqlite3
import pandas as pd
import pandas_ta as ta
import logging
import yfinance as yf
import json
import time
from datetime import date, timedelta
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.getcwd(), "data", "market_data.db")
CONFIG_DIR = os.path.join(os.getcwd(), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "watchlist.json")

class DatabaseManager:
    def __init__(self):
        """Initializes SQLite connection, ensures schema is up to date, and loads watchlist."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        self._create_tables()
        self._ensure_config_exists()
        self.watchlist = self._load_watchlist()

        self.ticker_aliases = {
            "TATAMOTORS": "TMPV"
        }

    def _create_tables(self):
        """Creates tables and safely migrates legacy schemas if columns are missing."""
        cursor = self.conn.cursor()
        
        # 1. OHLCV Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_ohlcv (
                ticker TEXT,
                timestamp DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
        
        # Safe migration: Add adj_close if it doesn't exist in legacy databases
        try:
            cursor.execute("ALTER TABLE daily_ohlcv ADD COLUMN adj_close REAL;")
            self.conn.commit()
            logger.info("Successfully migrated database schema: Added 'adj_close' column.")
        except sqlite3.OperationalError:
            # Column already exists, safe to ignore
            pass

        # 2. Sentiment Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_sentiment (
                ticker TEXT,
                timestamp DATE,
                sentiment_score REAL,
                sentiment_label TEXT,
                summary TEXT,
                PRIMARY KEY (ticker, timestamp)
            )
        """)

        # 3. Multi-Horizon Sentiment Table (1h, 1d, 1w)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_horizon_sentiment (
                ticker TEXT,
                timestamp DATETIME,
                sentiment_1h REAL,
                sentiment_1d REAL,
                sentiment_1w REAL,
                label_1h TEXT,
                label_1d TEXT,
                label_1w TEXT,
                divergence_flag TEXT,
                summary TEXT,
                headlines_count INTEGER,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
        self.conn.commit()

    def _ensure_config_exists(self):
        """Automatically creates config directory and watchlist.json if missing."""
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(CONFIG_DIR, exist_ok=True)
            default_watchlist = {
                "watchlist": [
                    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
                    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", 
                    "AXISBANK", "HINDUNILVR", "BAJFINANCE", "ASIANPAINT", "MARUTI",
                    "SUNPHARMA", "TITAN", "NTPC", "ONGC", "POWERGRID",
                    "TATASTEEL", "JSWSTEEL", "M&M", "ADANIENT", "ADANIPORTS",
                    "HCLTECH", "WIPRO", "BAJAJFINSV", "GRASIM", "TATACONSUM",
                    "BRITANNIA", "CIPLA", "DIVISLAB", "EICHERMOT", "HDFCLIFE",
                    "SBILIFE", "HINDZINC", "COALINDIA", "BPCL",
                    "IOC", "GAIL", "ULTRACEMCO", "TECHM", "NESTLEIND",
                    "DRREDDY", "APOLLOHOSP", "HEROMOTOCO", "BAJAJ-AUTO", "INDUSINDBK",
                    "SIEMENS", "ABB", "HAL", "BEL", "TVSMOTOR",
                    "CHOLAFIN", "SHRIRAMFIN", "ICICIPRULI", "SBICARD", "MUTHOOTFIN",
                    "SRF", "PIIND", "UPL", "AMBUJACEM", "GODREJCP",
                    "DABUR", "MARICO", "COLPAL", "PIDILITIND", "BERGEPAINT",
                    "VEDL", "HINDALCO", "NMDC", "JINDALSTEL", "POLICYBZR",
                    "ZOMATO", "PAYTM", "NYKAA", "LTIM", "PERSISTENT",
                    "MPHASIS", "COFORGE", "ICICIGI"
                ]
            }
            with open(CONFIG_PATH, "w") as f:
                json.dump(default_watchlist, f, indent=4)
            logger.info(f"Created missing config file at {CONFIG_PATH}")

    def _load_watchlist(self) -> list:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            watchlist = data.get("watchlist", [])
            logger.info(f"Loaded {len(watchlist)} tickers from {CONFIG_PATH}")
            return watchlist

    def seed_watchlist(self, days_back: int = 1825):
        """Fetches 5 years of historical data and stores it in SQLite."""
        to_date = date.today()
        from_date = to_date - timedelta(days=days_back)
        
        start_str = from_date.strftime("%Y-%m-%d")
        end_str = to_date.strftime("%Y-%m-%d")

        logger.info(f"Starting 5-year ingestion for {len(self.watchlist)} stocks...")

        for idx, ticker in enumerate(self.watchlist):
            query_ticker = self.ticker_aliases.get(ticker, ticker)
            yf_ticker = f"{query_ticker}.NS"
            
            time.sleep(1.5)
            
            try:
                df = yf.download(yf_ticker, start=start_str, end=end_str, progress=False, auto_adjust=False)
                
                if df.empty:
                    logger.warning(f"No data returned for {yf_ticker}. Skipping...")
                    continue
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                
                rename_map = {}
                for col in df.columns:
                    if 'date' in col or 'timestamp' in col:
                        rename_map[col] = 'timestamp'
                    elif 'open' in col:
                        rename_map[col] = 'open'
                    elif 'high' in col:
                        rename_map[col] = 'high'
                    elif 'low' in col:
                        rename_map[col] = 'low'
                    elif 'close' in col and 'adj' not in col:
                        rename_map[col] = 'close'
                    elif 'adj close' in col or 'adj_close' in col:
                        rename_map[col] = 'adj_close'
                    elif 'volume' in col:
                        rename_map[col] = 'volume'
                        
                df.rename(columns=rename_map, inplace=True)
                
                if 'adj_close' not in df.columns and 'close' in df.columns:
                    df['adj_close'] = df['close']

                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Columns mismatch for {ticker}. Skipping...")
                    continue

                df = df[required_cols].copy()
                df['ticker'] = ticker
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                df.dropna(inplace=True)
                
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM daily_ohlcv WHERE ticker = ?", (ticker,))
                self.conn.commit()

                df.to_sql("daily_ohlcv", self.conn, if_exists="append", index=False)
                logger.info(f"[{idx+1}/{len(self.watchlist)}] Successfully stored {len(df)} records for {ticker}.")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}. Skipping...")
                continue

    def get_connection(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)

    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        clean_ticker = ticker.split('.')[0].upper()
        query = f"SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv WHERE ticker = '{clean_ticker}' ORDER BY timestamp ASC"
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return df
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['return_1d'] = df['adj_close'].pct_change(1)
        df['return_3d'] = df['adj_close'].pct_change(3)
        df['return_5d'] = df['adj_close'].pct_change(5)
        
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def save_sentiment(self, ticker: str, timestamp: str, score: float, label: str, summary: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO stock_sentiment (ticker, timestamp, sentiment_score, sentiment_label, summary)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, timestamp, score, label, summary))
            conn.commit()

    def get_latest_sentiment(self, ticker: str) -> dict:
        query = f"SELECT timestamp, sentiment_score, sentiment_label, summary FROM stock_sentiment WHERE ticker = '{ticker}' ORDER BY timestamp DESC LIMIT 1"
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn)
        if df.empty:
            return {"sentiment_score": 0.0, "sentiment_label": "NEUTRAL", "summary": "No recent sentiment data."}
        return df.iloc[0].to_dict()

    def save_multi_horizon_sentiment(self, ticker: str, timestamp: str, s_1h: float, s_1d: float, s_1w: float,
                                     l_1h: str, l_1d: str, l_1w: str, divergence: str, summary: str, count: int):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO multi_horizon_sentiment 
                (ticker, timestamp, sentiment_1h, sentiment_1d, sentiment_1w, label_1h, label_1d, label_1w, divergence_flag, summary, headlines_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticker, timestamp, s_1h, s_1d, s_1w, l_1h, l_1d, l_1w, divergence, summary, count))
            conn.commit()

    def get_latest_multi_horizon_sentiment(self, ticker: str) -> dict:
        clean_ticker = ticker.split('.')[0].upper()
        query = f"SELECT * FROM multi_horizon_sentiment WHERE ticker = '{clean_ticker}' ORDER BY timestamp DESC LIMIT 1"
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn)
            if df.empty:
                return {
                    "sentiment_1h": 0.0, "label_1h": "NEUTRAL",
                    "sentiment_1d": 0.0, "label_1d": "NEUTRAL",
                    "sentiment_1w": 0.0, "label_1w": "NEUTRAL",
                    "divergence_flag": "NONE",
                    "summary": "No multi-horizon sentiment calculated yet.",
                    "headlines_count": 0,
                    "timestamp": None
                }
            return df.iloc[0].to_dict()
        except Exception:
            return {
                "sentiment_1h": 0.0, "label_1h": "NEUTRAL",
                "sentiment_1d": 0.0, "label_1d": "NEUTRAL",
                "sentiment_1w": 0.0, "label_1w": "NEUTRAL",
                "divergence_flag": "NONE",
                "summary": "No multi-horizon sentiment calculated yet.",
                "headlines_count": 0,
                "timestamp": None
            }


    def update_today_live_data(self, ticker: str):
        """Fetches today's live candle or latest price and updates SQLite."""
        query_ticker = self.ticker_aliases.get(ticker, ticker)
        yf_ticker = f"{query_ticker}.NS"
        
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            # Fetch the last 2 days with 1-day or 1-minute intervals to capture today's live state
            df = ticker_obj.history(period="2d", interval="1d", auto_adjust=False)
            
            if df.empty:
                return
                
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            
            # Map columns
            rename_map = {}
            for col in df.columns:
                if 'date' in col or 'timestamp' in col:
                    rename_map[col] = 'timestamp'
                elif 'open' in col:
                    rename_map[col] = 'open'
                elif 'high' in col:
                    rename_map[col] = 'high'
                elif 'low' in col:
                    rename_map[col] = 'low'
                elif 'close' in col and 'adj' not in col:
                    rename_map[col] = 'close'
                elif 'adj close' in col or 'adj_close' in col:
                    rename_map[col] = 'adj_close'
                elif 'volume' in col:
                    rename_map[col] = 'volume'
                    
            df.rename(columns=rename_map, inplace=True)
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close']

            df['ticker'] = ticker
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            
            # Take the absolute latest row (today's live bar)
            latest_row = df.iloc[[-1]].copy()
            required_cols = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            latest_row = latest_row[required_cols]
            
            cursor = self.conn.cursor()
            # Upsert today's candle so it stays up-to-date live
            cursor.execute("""
                INSERT OR REPLACE INTO daily_ohlcv (ticker, timestamp, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(latest_row.iloc[0]))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Could not fetch live data for {ticker}: {e}")
    
    def update_latest_data(self):
        """Pings yfinance for the last 5 days of data and upserts new candles into SQLite."""
        logger.info(f"Starting incremental daily update for {len(self.watchlist)} stocks...")

        for idx, ticker in enumerate(self.watchlist):
            query_ticker = self.ticker_aliases.get(ticker, ticker)
            yf_ticker = f"{query_ticker}.NS"
            
            time.sleep(1.0) # Rate-limiting pause
            
            try:
                # Fetch recent window (last 5 days) to ensure today's candle is captured
                df = yf.download(yf_ticker, period="5d", progress=False, auto_adjust=False)
                
                if df.empty:
                    continue
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                
                rename_map = {}
                for col in df.columns:
                    if 'date' in col or 'timestamp' in col:
                        rename_map[col] = 'timestamp'
                    elif 'open' in col:
                        rename_map[col] = 'open'
                    elif 'high' in col:
                        rename_map[col] = 'high'
                    elif 'low' in col:
                        rename_map[col] = 'low'
                    elif 'close' in col and 'adj' not in col:
                        rename_map[col] = 'close'
                    elif 'adj close' in col or 'adj_close' in col:
                        rename_map[col] = 'adj_close'
                    elif 'volume' in col:
                        rename_map[col] = 'volume'
                        
                df.rename(columns=rename_map, inplace=True)
                
                if 'adj_close' not in df.columns and 'close' in df.columns:
                    df['adj_close'] = df['close']

                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    continue

                df = df[required_cols].copy()
                df['ticker'] = ticker
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                df.dropna(inplace=True)
                
                cursor = self.conn.cursor()
                # Upsert records to prevent duplicates and append latest sessions
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_ohlcv (ticker, timestamp, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row['ticker'], row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['adj_close'], row['volume']))
                self.conn.commit()
                
                logger.info(f"[{idx+1}/{len(self.watchlist)}] Successfully updated latest records for {ticker}.")
                
            except Exception as e:
                logger.error(f"Error updating {ticker}: {e}")
                continue

if __name__ == "__main__":
    db = DatabaseManager()
    db.seed_watchlist(days_back=1825)


