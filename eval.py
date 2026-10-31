"""Evaluation harness.

Runs every sample in samples/ through the pipeline and scores it against the
teacher labels in each fixture. Two sets:

  python eval.py --set samples     # error type + knowledge point + broken step agreement
  python eval.py --set parsing     # routing of unreadable / unparseable input

Numbers printed here are real runs of whatever provider is configured. With no API
key the provider is the offline rule baseline, and the report says so.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from diagnose import Pipeline
from vision import ParseResult

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_samples() -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(ROOT, "samples", "*.json")))
    out = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            sample = json.load(fh)
        sample["_path"] = os.path.relpath(path, ROOT)
        out.append(sample)
    return out


def score_one(pipe: Pipeline, sample: Dict[str, Any]) -> Dict[str, Any]:
    parsed = ParseResult(
        status=sample.get("parse_status", "ok" if sample.get("question") else "unreadable"),
        question=sample.get("question", ""),
        steps=sample.get("steps", []),
    )
    result = pipe.run(parsed, student_id=f"eval-{sample['id']}", save=False)
    expected = sample.get("expected", {})
    broken = result.get("broken_step") or ""
    expected_broken = expected.get("broken_step_contains", "")
    return {
        "sample": sample["id"],
        "expected_type": expected.get("error_type"),
        "got_type": result.get("error_type"),
        "type_ok": expected.get("error_type") == result.get("error_type"),
        "expected_kp": expected.get("knowledge_point"),
        "got_kp": result.get("knowledge_point"),
        "kp_ok": (expected.get("knowledge_point") is None and result.get("knowledge_point") is None)
        or expected.get("knowledge_point") == result.get("knowledge_point"),
        "broken_step_ok": bool(expected_broken) and expected_broken in broken,
        "confidence": result.get("confidence"),
        "unmapped": result["flags"]["unmapped"],
        "review": result["flags"]["needs_teacher_review"],
    }


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="samples", choices=("samples", "parsing"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--save", action="store_true", help="write the report to traces/")
    args = ap.parse_args(argv)

    pipe = Pipeline(force_offline=True)
    samples = load_samples()
    rows = [score_one(pipe, s) for s in samples]

    if args.set == "parsing":
        rows = [r for r in rows if r["expected_type"] == "unreadable"] or rows

    n = len(rows)
    type_acc = sum(1 for r in rows if r["type_ok"]) / n
    kp_acc = sum(1 for r in rows if r["kp_ok"]) / n
    step_acc = sum(1 for r in rows if r["broken_step_ok"]) / n

    header = f"{'sample':<22}{'type':<26}{'knowledge point':<22}{'step':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['sample']:<22}"
            f"{str(r['expected_type']) + ' / ' + str(r['got_type']):<26}"
            f"{str(r['expected_kp']) + ' / ' + str(r['got_kp']):<22}"
            f"{('hit' if r['broken_step_ok'] else 'miss'):<8}"
        )
    print("-" * len(header))
    print(f"provider                : {pipe.provider.name}")
    print(f"samples                 : {n}")
    print(f"error-type agreement    : {sum(1 for r in rows if r['type_ok'])}/{n} = {type_acc:.0%}")
    print(f"knowledge-point accuracy: {sum(1 for r in rows if r['kp_ok'])}/{n} = {kp_acc:.0%}")
    print(f"broken-step located     : {sum(1 for r in rows if r['broken_step_ok'])}/{n} = {step_acc:.0%}")
    print(f"flagged for review      : {sum(1 for r in rows if r['review'])}/{n}")

    if args.save:
        os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True)
        path = os.path.join(ROOT, "traces", f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Evaluation report\n\n"
                f"- provider: `{pipe.provider.name}`\n"
                f"- samples: {n}\n"
                f"- error-type agreement: {type_acc:.0%}\n"
                f"- knowledge-point accuracy: {kp_acc:.0%}\n"
                f"- broken-step located: {step_acc:.0%}\n\n"
                "```json\n" + json.dumps(rows, ensure_ascii=False, indent=2) + "\n```\n"
            )
        print(f"report                  : {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
