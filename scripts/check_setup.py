"""Read-only local diagnostics: never prints environment secrets or trains models."""
import importlib.util
import json
from pathlib import Path
import platform


def main():
    root = Path(__file__).resolve().parents[1]
    modules = ["numpy", "pandas", "pandas_ta", "yfinance", "streamlit", "joblib",
               "torch", "sklearn", "langchain_ollama", "langchain_chroma", "langgraph"]
    print(json.dumps({
        "python": platform.python_version(), "project_root": str(root),
        "modules_discoverable": {name: importlib.util.find_spec(name) is not None for name in modules},
        "artifacts_present": {name: (root / name).exists() for name in [
            "config/watchlist.json", "data/market_data.db", "data/models/best_model.pkl",
            "data/models/deep_learning_lstm.pt", "data/chromadb"]},
        "execution_mode": "Research only; no broker adapter/order placement implemented",
        "note": "Module discovery does not verify imports, compatibility, data freshness or model quality"
    }, indent=2))


if __name__ == "__main__":
    main()