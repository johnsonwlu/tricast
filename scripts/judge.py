"""Score the analyst LLM's output with an LLM judge, across tickers.

The judge should be a *stronger* model than the analyst (grading your own output
is biased). Default judge is anthropic (needs ANTHROPIC_API_KEY); point it at a
different local model with --judge-provider ollama for a key-free run.

Examples:
  python scripts/judge.py AAPL MSFT NVDA
  python scripts/judge.py AAPL --analyst-provider ollama --judge-provider ollama
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from tricast import judge  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--analyst-provider", default="ollama")
    p.add_argument("--judge-provider", default="anthropic")
    p.add_argument("--json", action="store_true", help="dump raw rows as JSON")
    args = p.parse_args()

    rows = judge.run_judge_panel(
        args.tickers, analyst_provider=args.analyst_provider,
        judge_provider=args.judge_provider,
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    print("\n" + judge.format_summary(judge.summarize_judge(rows)))


if __name__ == "__main__":
    main()
