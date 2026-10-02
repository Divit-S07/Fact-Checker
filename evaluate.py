"""
Optional enhancement: evaluate verdict accuracy on a labelled set of known claims.

Usage:
    python evaluate.py            # full set (12 claims, ~3-5 min)
    python evaluate.py 5          # first 5 claims only
"""

import json
import sys
import time
from pathlib import Path

from agent_pipeline import run_fact_check

LABELLED = Path(__file__).parent / "data" / "labelled_claims.json"


def evaluate(limit: int | None = None) -> dict:
    dataset = json.loads(LABELLED.read_text(encoding="utf-8"))
    if limit:
        dataset = dataset[:limit]

    rows = []
    correct = 0
    t0 = time.time()

    for i, item in enumerate(dataset, 1):
        claim, expected = item["claim"], item["expected"]
        t = time.time()
        state = run_fact_check(claim)
        results = state.get("claim_results", [])
        verdict = results[0]["verdict"] if results else "Unverifiable"
        hit = verdict == expected
        correct += int(hit)
        rows.append({
            "claim": claim,
            "expected": expected,
            "predicted": verdict,
            "confidence": round(results[0]["confidence"], 2) if results else 0,
            "match": hit,
            "seconds": round(time.time() - t, 1),
        })
        print(f"[{i}/{len(dataset)}] {'PASS' if hit else 'FAIL'} "
              f"expected={expected} got={verdict} ({rows[-1]['seconds']}s) | {claim[:60]}")

    accuracy = correct / len(rows) if rows else 0.0
    report = {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": len(rows),
        "elapsed_seconds": round(time.time() - t0, 1),
        "confusion_notes": "Exact-match accuracy; Unverifiable predictions count as correct only when expected.",
        "results": rows,
    }

    out = Path(__file__).parent / "data" / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAccuracy: {correct}/{len(rows)} = {accuracy:.0%}  "
          f"({report['elapsed_seconds']}s) -> {out}")
    return report


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evaluate(n)
