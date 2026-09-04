import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sqlite3
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.getcwd(), "data", "market_data.db")
MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deep_learning_lstm.pt")

class PriceSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class BiLSTMTrajectoryModel(nn.Module):
    """
    Bidirectional LSTM with residual projection for predicting multi-step price envelope:
    Outputs: [expected_return_close, expected_return_high, expected_return_low]
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3) # Targets: [close_ret, high_ret, low_ret]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x) # (B, Seq_len, hidden*2)
        last_step = lstm_out[:, -1, :] # Take final time-step representation
        out = self.fc(last_step)
        return out

class DeepLearningPredictor:
    """
    Deep Learning sequence predictor for intraday/short-term price trajectories.
    Uses 30-period lookback sequences to predict next 3-period High, Low, and Close bounds.
    """
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        self.input_dim = 8 # open, high, low, close, volume, return_1d, volatility_14, rsi_14
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMTrajectoryModel(input_dim=self.input_dim).to(self.device)
        self.is_trained = False

        os.makedirs(MODEL_DIR, exist_ok=True)
        if os.path.exists(MODEL_PATH):
            self.load_model()

    def load_model(self):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_trained = True
            logger.info("Deep learning LSTM weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load LSTM weights: {e}")

    def save_model(self):
        torch.save(self.model.state_dict(), MODEL_PATH)
        logger.info(f"Model successfully saved to {MODEL_PATH}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates normalized sequential features."""
        df = df.copy()
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']

        df['return_1d'] = df['adj_close'].pct_change().fillna(0.0)
        df['volatility_14'] = df['return_1d'].rolling(window=14, min_periods=1).std().fillna(0.01)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-5)
        df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50.0)

        # Normalize relative to rolling base
        rolling_close = df['close'].rolling(window=self.seq_len, min_periods=1).mean()
        df['norm_open'] = (df['open'] - rolling_close) / (rolling_close + 1e-5)
        df['norm_high'] = (df['high'] - rolling_close) / (rolling_close + 1e-5)
        df['norm_low'] = (df['low'] - rolling_close) / (rolling_close + 1e-5)
        df['norm_close'] = (df['close'] - rolling_close) / (rolling_close + 1e-5)

        rolling_vol = df['volume'].rolling(window=self.seq_len, min_periods=1).mean()
        df['norm_volume'] = (df['volume'] - rolling_vol) / (rolling_vol + 1e-5)

        # Normalization for RSI and volatility
        df['norm_rsi'] = (df['rsi_14'] - 50.0) / 25.0
        df['norm_volat'] = (df['volatility_14'] - 0.015) / 0.015

        return df

    def create_dataset_from_db(self, limit_stocks: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Gathers sequential training pairs from SQLite database."""
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT DISTINCT ticker FROM daily_ohlcv LIMIT ?"
        tickers_df = pd.read_sql(query, conn, params=(limit_stocks,))

        sequences = []
        targets = []

        feature_cols = [
            'norm_open', 'norm_high', 'norm_low', 'norm_close',
            'norm_volume', 'return_1d', 'norm_volat', 'norm_rsi'
        ]

        for ticker in tickers_df['ticker']:
            stock_query = f"SELECT timestamp, open, high, low, close, volume FROM daily_ohlcv WHERE ticker = '{ticker}' ORDER BY timestamp ASC"
            df = pd.read_sql(stock_query, conn)
            if len(df) < (self.seq_len + 15):
                continue

            df = self.prepare_features(df)
            feature_matrix = df[feature_cols].values
            close_vals = df['close'].values
            high_vals = df['high'].values
            low_vals = df['low'].values

            for i in range(self.seq_len, len(df) - 3):
                seq = feature_matrix[i - self.seq_len:i]
                curr_close = close_vals[i - 1]

                # Future 3-step returns
                future_close = close_vals[i + 2]
                future_high = np.max(high_vals[i:i + 3])
                future_low = np.min(low_vals[i:i + 3])

                ret_close = (future_close - curr_close) / curr_close
                ret_high = (future_high - curr_close) / curr_close
                ret_low = (future_low - curr_close) / curr_close

                # Clip extreme anomalies
                ret_close = np.clip(ret_close, -0.15, 0.15)
                ret_high = np.clip(ret_high, -0.15, 0.20)
                ret_low = np.clip(ret_low, -0.20, 0.15)

                sequences.append(seq)
                targets.append([ret_close, ret_high, ret_low])

        conn.close()
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def train(self, epochs: int = 15, batch_size: int = 64, lr: float = 0.001):
        """Trains the Bi-LSTM model on historical sequential windows."""
        logger.info("Extracting sequential training sequences from SQLite...")
        X, y = self.create_dataset_from_db()

        if len(X) == 0:
            logger.warning("No sequences extracted. Ensure market_data.db has historical data.")
            return

        logger.info(f"Loaded {len(X)} training sequences of length {self.seq_len}.")
        dataset = PriceSequenceDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_x)

            epoch_loss = total_loss / len(dataset)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Huber Loss: {epoch_loss:.6f}")

        self.is_trained = True
        self.save_model()

    def predict_trajectory(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs inference on recent price sequence to output expected price envelope.
        """
        if df.empty or len(df) < self.seq_len:
            latest_price = float(df['close'].iloc[-1]) if not df.empty else 100.0
            return {
                "predicted_close": latest_price,
                "predicted_high": latest_price * 1.015,
                "predicted_low": latest_price * 0.985,
                "expected_return_pct": 0.0,
                "volatility_spread_pct": 3.0,
                "trajectory_bias": "NEUTRAL",
                "confidence": 50.0
            }

        df_feat = self.prepare_features(df)
        feature_cols = [
            'norm_open', 'norm_high', 'norm_low', 'norm_close',
            'norm_volume', 'return_1d', 'norm_volat', 'norm_rsi'
        ]
        seq_data = df_feat[feature_cols].iloc[-self.seq_len:].values

        if np.isnan(seq_data).any():
            seq_data = np.nan_to_num(seq_data, nan=0.0)

        seq_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(seq_tensor).cpu().numpy()[0]

        ret_close, ret_high, ret_low = preds[0], preds[1], preds[2]

        latest_close = float(df['close'].iloc[-1])
        predicted_close = round(latest_close * (1.0 + float(ret_close)), 2)
        predicted_high = round(latest_close * (1.0 + max(float(ret_high), float(ret_close))), 2)
        predicted_low = round(latest_close * (1.0 + min(float(ret_low), float(ret_close))), 2)

        ret_pct = round(float(ret_close) * 100, 2)
        spread_pct = round(((predicted_high - predicted_low) / latest_close) * 100, 2)

        bias = "BULLISH" if ret_pct > 0.3 else ("BEARISH" if ret_pct < -0.3 else "NEUTRAL")
        confidence = min(95.0, max(50.0, 50.0 + abs(ret_pct) * 8.0))

        return {
            "predicted_close": predicted_close,
            "predicted_high": predicted_high,
            "predicted_low": predicted_low,
            "expected_return_pct": ret_pct,
            "volatility_spread_pct": spread_pct,
            "trajectory_bias": bias,
            "confidence": round(confidence, 1)
        }

if __name__ == "__main__":
    predictor = DeepLearningPredictor()
    if not predictor.is_trained:
        print("Training Bi-LSTM Trajectory Model...")
        predictor.train(epochs=10)

    # Test inference on sample stock
    conn = sqlite3.connect(DB_PATH)
    test_df = pd.read_sql("SELECT timestamp, open, high, low, close, volume FROM daily_ohlcv WHERE ticker = 'RELIANCE' ORDER BY timestamp ASC", conn)
    conn.close()

    if not test_df.empty:
        trajectory = predictor.predict_trajectory(test_df)
        print("\n--- Deep Learning Trajectory Prediction (RELIANCE) ---")
        for k, v in trajectory.items():
            print(f"{k}: {v}")
