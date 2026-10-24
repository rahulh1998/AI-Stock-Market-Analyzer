"""Measure the decision model against the regex matcher on news relevance.

This is the question that decides whether the LLM layer earns its place: the alias matcher
is fast, free and deterministic, so a model is only worth calling if it is measurably more
accurate on the cases that matter.

The labelled set below is built from headlines actually observed on Google News for these
queries, including both failure modes seen in development:
  - false positives the regex lets through (a market round-up naming two companies)
  - false negatives the regex drops ("Reliance Jio Platforms IPO", which no alias matches)

Run:  python -m scripts.eval_relevance
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

from src.llm_engine.providers import build_provider
from src.llm_engine.tasks import DecisionCache, DecisionEngine
from src.sentiment_engine.news_sources import EntityMatcher, load_instruments

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# (headline, ticker, company, is_really_about_this_company)
LABELLED = [
    # --- sibling companies: the regex needs a curated alias list to survive these ---
    ("Reliance Power shares surge 8% on new order win", "RELIANCE", "Reliance Industries", False),
    ("Reliance Infrastructure board approves fundraise", "RELIANCE", "Reliance Industries", False),
    ("Adani Power wins 1,600 MW supply order", "ADANIENT", "Adani Enterprises", False),
    ("Adani Green Energy commissions solar plant", "ADANIENT", "Adani Enterprises", False),
    ("Bajaj Housing Finance lists at a premium", "BAJFINANCE", "Bajaj Finance", False),
    ("SBI Cards and Payment Services Q2 profit rises 12%", "SBIN", "State Bank of India", False),
    ("SBI Life Insurance reports strong APE growth", "SBIN", "State Bank of India", False),
    ("Tata Motors Finance merges with a group entity", "TATASTEEL", "Tata Steel", False),
    # --- genuine company news ---
    ("Reliance Industries shares jump 3% as Jio IPO fully subscribed", "RELIANCE", "Reliance Industries", True),
    ("RIL rises 3% as Jio IPO date announced", "RELIANCE", "Reliance Industries", True),
    ("Reliance Jio Platforms IPO sees strong institutional demand", "RELIANCE", "Reliance Industries", True),
    ("State Bank of India cuts MCLR by 10 basis points", "SBIN", "State Bank of India", True),
    ("SBI raises Rs 10,000 crore via infrastructure bonds", "SBIN", "State Bank of India", True),
    ("Tata Steel slumps 6% after Q2 loss widens on weak demand", "TATASTEEL", "Tata Steel", True),
    ("Tata Steel Netherlands announces restructuring plan", "TATASTEEL", "Tata Steel", True),
    ("Infosys raises FY27 revenue guidance after strong deal wins", "INFY", "Infosys", True),
    ("Sebi opens probe into disclosure lapses at Adani Enterprises", "ADANIENT", "Adani Enterprises", True),
    ("Bajaj Finance AUM grows 28% in the September quarter", "BAJFINANCE", "Bajaj Finance", True),
    # --- market round-ups: about nobody in particular ---
    ("Top stocks to watch today: Eternal, Reliance Industries, IREDA in focus", "RELIANCE", "Reliance Industries", False),
    ("Stock markets trim early losses amid buying in SBI, Reliance Industries", "RELIANCE", "Reliance Industries", False),
    ("Sensex, Nifty rebound after three-day fall on IT stocks buying", "INFY", "Infosys", False),
    ("Nifty metal index gains 2% as Tata Steel, JSW Steel advance", "TATASTEEL", "Tata Steel", False),
    # --- boilerplate quote pages ---
    ("Reliance Industries Share Price - Live NSE Quote", "RELIANCE", "Reliance Industries", False),
    ("Tata Steel Ltd (TATASTEEL.NS) Stock Price, News, Quote & History", "TATASTEEL", "Tata Steel", False),
]


def score(predictions, truth) -> dict:
    tp = sum(1 for p, t in zip(predictions, truth) if p and t)
    tn = sum(1 for p, t in zip(predictions, truth) if not p and not t)
    fp = sum(1 for p, t in zip(predictions, truth) if p and not t)
    fn = sum(1 for p, t in zip(predictions, truth) if not p and t)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(truth), "precision": precision, "recall": recall,
            "f1": f1, "true_pos": tp, "true_neg": tn, "false_pos": fp, "false_neg": fn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="auto", choices=["auto", "ollama", "jev", "none"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--veto-confidence", type=float, default=0.6,
                        help="Model confidence needed before it may veto a regex match")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Below this the model abstains and the regex answer is used")
    args = parser.parse_args()

    truth = [row[3] for row in LABELLED]
    instruments = load_instruments()
    matcher = EntityMatcher(instruments)

    print(f"Labelled headlines: {len(LABELLED)}  "
          f"(about-company: {sum(truth)}, not: {len(truth) - sum(truth)})\n")

    regex_predictions = [matcher.attribute(h, t)[0] for h, t, _, _ in LABELLED]
    regex = score(regex_predictions, truth)

    provider = build_provider(args.provider, args.model)
    engine = DecisionEngine(provider, DecisionCache(enabled=not args.no_cache))
    print(f"Provider: {provider.name} {getattr(provider, 'model', '')}  "
          f"(available={provider.available()})\n")

    llm_predictions, details = [], []
    started = time.time()
    for headline, ticker, company, _ in LABELLED:
        decision = engine.classify_relevance(headline, company, ticker)
        if not decision.available or decision.confidence < args.min_confidence:
            predicted = matcher.attribute(headline, ticker)[0]
            source = "regex(fallback)"
        else:
            predicted = decision.label == "about_company"
            source = decision.label
        llm_predictions.append(predicted)
        details.append((headline, ticker, predicted, decision, source))
    elapsed = time.time() - started
    model = score(llm_predictions, truth)

    print(f"{'':<4}{'headline':<62}{'truth':<7}{'regex':<7}{'llm':<7}  label")
    print("-" * 110)
    for index, ((headline, ticker, _, actual), regex_prediction, (_, _, llm_prediction, decision, source)) \
            in enumerate(zip(LABELLED, regex_predictions, details)):
        flag = "  " if regex_prediction == llm_prediction else ("++" if llm_prediction == actual else "--")
        print(f"{flag:<4}{headline[:60]:<62}{str(actual):<7}{str(regex_prediction):<7}"
              f"{str(llm_prediction):<7}  {source}")

    # Hybrid: the regex decides, and the model may only REMOVE a headline it judges a
    # market-wide round-up. Same invariant as the trading engine — the model vetoes, never
    # creates. The two methods fail on different cases, so this should beat both.
    hybrid_predictions = []
    for regex_prediction, (_, _, _, decision, _) in zip(regex_predictions, details):
        vetoed = (decision.available and decision.label == "market_wide"
                  and decision.confidence >= args.veto_confidence)
        hybrid_predictions.append(regex_prediction and not vetoed)
    hybrid = score(hybrid_predictions, truth)

    print("\n" + "=" * 84)
    print(f"{'metric':<14}{'regex':>10}{'model':>10}{'hybrid':>10}{'hyb-regex':>12}")
    print("-" * 84)
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key:<14}{regex[key]:>10.3f}{model[key]:>10.3f}{hybrid[key]:>10.3f}"
              f"{hybrid[key] - regex[key]:>+12.3f}")
    for key in ("false_pos", "false_neg"):
        print(f"{key:<14}{regex[key]:>10d}{model[key]:>10d}{hybrid[key]:>10d}"
              f"{hybrid[key] - regex[key]:>+12d}")
    print("=" * 84)
    print(f"model latency: {elapsed:.1f}s total, {elapsed / len(LABELLED):.2f}s/headline "
          f"(cache hits {engine.stats['cache_hits']}, calls {engine.stats['calls']}, "
          f"failures {engine.stats['failures']})")

    best = max((regex["f1"], "regex alone"), (model["f1"], "model alone"),
               (hybrid["f1"], "hybrid (regex decides, model vetoes round-ups)"))
    print(f"\nBEST: {best[1]} at F1 {best[0]:.3f}")
    if hybrid["f1"] > max(regex["f1"], model["f1"]):
        print("VERDICT: the veto hybrid earns its place — the model is called, but only to remove.")
    elif model["f1"] > regex["f1"]:
        print("VERDICT: the model beats the regex outright.")
    else:
        print("VERDICT: nothing beats the regex. Keep it; it is free and deterministic.")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
