import pandas as pd
import pandas_ta as ta
import logging
from datetime import date, timedelta
from jugaad_data.nse import stock_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self):
        """
        Initializes the data fetcher using official NSE India endpoints.
        """
        pass

    def fetch_historical_data(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical OHLCV data directly from NSE.
        Automatically strips '.NS' suffixes to match NSE formatting.
        """
        # Clean ticker for NSE (e.g., 'TATAMOTORS.NS' -> 'TATAMOTORS')
        clean_ticker = ticker.split('.')[0]
        
        # Convert 'period' string to days
        days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        period_days = days_map.get(period, 180)
        
        logger.info(f"Fetching {period_days} days of data for {clean_ticker} from NSE directly...")
        
        try:
            to_date = date.today()
            from_date = to_date - timedelta(days=period_days)
            
            # Fetch data directly from NSE India (Equity series)
            df = stock_df(symbol=clean_ticker, from_date=from_date, to_date=to_date, series="EQ")
            
            if df.empty:
                logger.warning(f"No data returned for {clean_ticker}. Check if the symbol is correct.")
                return pd.DataFrame()
                
            # NSE returns data newest-first. Reverse it for technical indicators.
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Standardize column names for downstream agents
            df.rename(columns={
                'DATE': 'timestamp',
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'VOLUME': 'volume'
            }, inplace=True)
            
            # Filter out irrelevant NSE metadata
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Successfully fetched {len(df)} rows for {clean_ticker}.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from NSE for {ticker}: {e}")
            return pd.DataFrame()

    def enrich_with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies technical indicators using pandas-ta.
        """
        if df.empty:
            return df
            
        logger.info("Calculating technical indicators...")
        
        df.ta.ema(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    def get_latest_features(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        
        latest = df.iloc[-1].to_dict()
        latest['timestamp'] = str(latest['timestamp'])
        return latest

if __name__ == "__main__":
    # Quick test execution
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data(ticker="TATAMOTORS.NS", period="6mo")
    
    if not df.empty:
        enriched_df = fetcher.enrich_with_indicators(df)
        print("\nLatest Technical Snapshot:")
        latest_data = fetcher.get_latest_features(enriched_df)
        for k, v in latest_data.items():
            print(f"{k}: {v}")