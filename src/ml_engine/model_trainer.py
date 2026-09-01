import os
import sqlite3
import logging
import json
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.getcwd(), "data", "market_data.db")
CONFIG_PATH = os.path.join(os.getcwd(), "config", "nifty100_watchlist.json")
MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

class StockModelTrainer:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.watchlist = self._load_watchlist()

    def _load_watchlist(self) -> list:
        if not os.path.exists(CONFIG_PATH):
            return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "TATAMOTORS"]
        with open(CONFIG_PATH, "r") as f:
            return json.load(f).get("watchlist", [])

    def load_all_data(self) -> pd.DataFrame:
        """Pulls historical data for all tickers from SQLite and computes features."""
        all_dfs = []
        
        for ticker in self.watchlist:
            query = f"SELECT timestamp, open, high, low, close, volume FROM daily_ohlcv WHERE ticker = '{ticker}' ORDER BY timestamp ASC"
            df = pd.read_sql(query, self.conn)
            
            if df.empty or len(df) < 200:
                continue
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # --- Feature Engineering ---
            # 1. Percentage Returns & Momentum
            df['pct_change'] = df['close'].pct_change()
            df['volatility'] = df['pct_change'].rolling(window=14).std()
            
            # 2. Moving Average Distances
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            df['dist_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
            df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
            df['dist_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
            
            # 3. RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # 4. Target Variable: 1 if tomorrow's close > today's close, else 0
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            df.dropna(inplace=True)
            df['ticker'] = ticker
            all_dfs.append(df)
            
        if not all_dfs:
            raise ValueError("No data available to train model.")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded dataset with {len(combined_df)} total records across watchlist.")
        return combined_df

    def train(self):
        """Trains the XGBoost Classifier using chronological time-series splitting."""
        df = self.load_all_data()
        
        # Select feature columns
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'pct_change', 'volatility', 
            'dist_ema20', 'dist_sma50', 'dist_sma200', 'rsi_14'
        ]
        
        X = df[feature_cols]
        y = df['target']
        
        # Chronological Train/Test Split (No random shuffling to prevent time-series data leakage!)
        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        logger.info(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")
        
        # Initialize XGBoost Classifier
        model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        logger.info("Training XGBoost model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Model Test Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, preds))
        
        # Save trained model to disk
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Trained model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    trainer = StockModelTrainer()
    trainer.train()