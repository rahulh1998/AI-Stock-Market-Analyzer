import sqlite3
import pandas as pd
import pandas_ta as ta
import logging
import requests
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
        """Initializes SQLite connection, tables, ensures config exists, and loads watchlist."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self._create_tables()
        self._ensure_config_exists()
        self.watchlist = self._load_watchlist()

        self.ticker_aliases = {
            "TATAMOTORS": "TMPV"
        }

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        })

    def _ensure_config_exists(self):
        """Automatically creates the config directory and watchlist.json file if missing."""
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

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_ohlcv (
                ticker TEXT,
                timestamp DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
        self.conn.commit()

    def seed_watchlist(self, days_back: int = 1825):
        to_date = date.today()
        from_date = to_date - timedelta(days=days_back)
        
        start_str = from_date.strftime("%Y-%m-%d")
        end_str = to_date.strftime("%Y-%m-%d")

        logger.info(f"Starting 5-year ingestion for {len(self.watchlist)} stocks...")

        for ticker in self.watchlist:
            query_ticker = self.ticker_aliases.get(ticker, ticker)
            yf_ticker = f"{query_ticker}.NS"
            
            time.sleep(1.0)
            
            try:
                ticker_obj = yf.Ticker(yf_ticker, session=self.session)
                df = ticker_obj.history(start=start_str, end=end_str, auto_adjust=False)
                
                if df.empty:
                    logger.warning(f"No data returned for {yf_ticker}. Skipping...")
                    continue
                    
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
                logger.info(f"Successfully stored {len(df)} records for {ticker}.")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}. Skipping...")
                continue

    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        clean_ticker = ticker.split('.')[0].upper()
        query = f"SELECT timestamp, open, high, low, close, adj_close, volume FROM daily_ohlcv WHERE ticker = '{clean_ticker}' ORDER BY timestamp ASC"
        df = pd.read_sql(query, self.conn)
        
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

if __name__ == "__main__":
    db = DatabaseManager()
    db.seed_watchlist(days_back=1825)