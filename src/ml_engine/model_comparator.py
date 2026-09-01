import os
import sqlite3
import logging
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.getcwd(), "data", "market_data.db")
CONFIG_PATH = os.path.join(os.getcwd(), "config", "nifty100_watchlist.json")
MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

class ModelComparator:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.watchlist = self._load_watchlist()

    def _load_watchlist(self) -> list:
        if not os.path.exists(CONFIG_PATH):
            return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "TATAMOTORS"]
        with open(CONFIG_PATH, "r") as f:
            return json.load(f).get("watchlist", [])

    def load_and_engineer_features(self) -> pd.DataFrame:
        """
        Engineers technical features and strictly lags them by 1 trading day 
        to eliminate look-ahead bias and data leakage.
        """
        all_dfs = []
        
        for ticker in self.watchlist:
            query = f"SELECT timestamp, open, high, low, close, volume FROM daily_ohlcv WHERE ticker = '{ticker}' ORDER BY timestamp ASC"
            df = pd.read_sql(query, self.conn)
            
            if df.empty or len(df) < 250:
                continue
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # --- 1. Raw Calculations (Before Shifting) ---
            df['return_1d'] = df['close'].pct_change(1)
            df['return_3d'] = df['close'].pct_change(3)
            df['return_5d'] = df['close'].pct_change(5)
            
            df['volatility_14'] = df['return_1d'].rolling(window=14).std()
            df['intraday_spread'] = (df['high'] - df['low']) / df['close']
            
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
            df['vol_ratio'] = df['volume'] / (df['vol_ma_20'] + 1e-5)

            # --- 2. CRITICAL: Lag all features and indicators by 1 trading day ---
            feature_cols_raw = [
                'close', 'return_1d', 'return_3d', 'return_5d',
                'volatility_14', 'intraday_spread',
                'ema_9', 'ema_21', 'sma_50', 'sma_200',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'vol_ratio'
            ]
            
            for col in feature_cols_raw:
                df[f'{col}_lag1'] = df[col].shift(1)
            
            df['dist_ema9_lag1'] = (df['close_lag1'] - df['ema_9_lag1']) / df['ema_9_lag1']
            df['dist_sma50_lag1'] = (df['close_lag1'] - df['sma_50_lag1']) / df['sma_50_lag1']

            # --- Target: 3-Day Forward Trend Direction ---
            df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
            
            df.dropna(inplace=True)
            df['ticker'] = ticker
            all_dfs.append(df)
            
        if not all_dfs:
            raise ValueError("Insufficient data available in SQLite database.")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Engineered leakage-free lagged features. Total records: {len(combined_df)}")
        return combined_df

    def evaluate_models(self):
        df = self.load_and_engineer_features()
        
        # Strictly use lagged feature vectors to prevent look-ahead bias
        feature_cols = [
            'close_lag1', 'return_1d_lag1', 'return_3d_lag1', 'return_5d_lag1',
            'volatility_14_lag1', 'intraday_spread_lag1',
            'dist_ema9_lag1', 'dist_sma50_lag1',
            'rsi_14_lag1', 'macd_lag1', 'macd_signal_lag1', 'macd_hist_lag1', 'vol_ratio_lag1'
        ]
        
        X = df[feature_cols]
        y = df['target']

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
            "XGBoost Classifier": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, eval_metric='logloss'),
            "LightGBM Classifier": LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1),
            "CatBoost Classifier": CatBoostClassifier(iterations=150, depth=4, learning_rate=0.03, random_seed=42, verbose=0),
            "Support Vector Machine": SVC(probability=True, random_state=42)
        }

        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {}

        print("\n" + "="*55)
        print("  LEAKAGE-FREE WALK-FORWARD BENCHMARK RESULTS")
        print("="*55)

        for name, model in models.items():
            accuracies = []
            precisions = []
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                accuracies.append(accuracy_score(y_test, preds))
                precisions.append(precision_score(y_test, preds, zero_division=0))
                
            mean_acc = np.mean(accuracies) * 100
            mean_prec = np.mean(precisions) * 100
            
            model_scores[name] = {
                "model_obj": model,
                "accuracy": mean_acc,
                "precision": mean_prec
            }
            print(f"🔹 {name}")
            print(f"   Mean Out-of-Sample Accuracy:  {mean_acc:.2f}%")
            print(f"   Mean Out-of-Sample Precision: {mean_prec:.2f}%\n")

        best_model_name = max(model_scores, key=lambda k: model_scores[k]["precision"])
        best_model = model_scores[best_model_name]["model_obj"]

        print("="*55)
        print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
        print("="*55)

        logger.info(f"Retraining winning model on complete dataset...")
        best_model.fit(X, y)
        
        joblib.dump(best_model, BEST_MODEL_PATH)
        logger.info(f"Model serialized and saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.evaluate_models()