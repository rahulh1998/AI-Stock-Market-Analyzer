"""Project locations resolved from this file, never from the working directory.

Existing modules built their paths with os.path.join(os.getcwd(), ...), so the app
only worked when launched from the project root. Importing from here removes that
constraint without changing any on-disk layout.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
BOOKS_DIR = DATA_DIR / "books"
CHROMA_DIR = DATA_DIR / "chromadb"

DB_PATH = DATA_DIR / "market_data.db"
WATCHLIST_PATH = CONFIG_DIR / "watchlist.json"
INSTRUMENTS_PATH = CONFIG_DIR / "instruments.json"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"


def ensure_data_dirs() -> None:
    """Create the writable data directories. Safe to call repeatedly."""
    for directory in (DATA_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
