"""Core pipeline: parse → diagnose → ground → prescribe → schedule → store.

Also the CLI adapter:
    python diagnose.py --sample samples/alg_distributive.json --trace
    python diagnose.py --question "Solve x - 4 = 10" --steps "x = 10 - 4;x = 6"
    python diagnose.py --image photo.jpg --trace        # needs a vision provider
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from grounding import Grounder, load_kb
from llm import Config, ERROR_TYPES, LLMError, get_provider
from schedule import Gap, build_path, initial_path
from store import add_mistake, connect, gap_profile, init_db
from vision import RETAKE_MESSAGE, ParseResult, from_text, parse_photo

ROOT = os.path.dirname(os.path.abspath(__file__))


class Pipeline:
    def __init__(self, cfg: Optional[Config] = None, provider=None, grounder=None, force_offline: bool = False):
        self.cfg = cfg or Config.from_env()
        self.provider = provider or get_provider(self.cfg, force_offline=force_offline)
        if grounder is None:
            nodes, matchers = load_kb(
                os.path.join(ROOT, self.cfg.kb_path), os.path.join(ROOT, self.cfg.matchers_path)
            )
            self.nodes = nodes
            self.grounder = Grounder(nodes, matchers, self.provider, threshold=self.cfg.grounding_threshold)
        else:
            self.grounder = grounder
            self.nodes = grounder.nodes
        self.trace: List[Dict[str, Any]] = []

    def _step(self, name: str, payload: Any) -> None:
        self.trace.append({"step": name, "at": datetime.utcnow().isoformat(timespec="seconds"), "payload": payload})

    def node_by_id(self, kp_id: str) -> Optional[Dict[str, Any]]:
        return next((n for n in self.nodes if n["id"] == kp_id), None)

    def run(
        self,
        parsed: ParseResult,
        student_id: str = "anon",
        image_path: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        self.trace = []
        self._step("parse", parsed.to_dict())

        if not parsed.ok:
            self._step("abort", {"reason": parsed.status, "message": RETAKE_MESSAGE})
            return {
                "status": parsed.status,
                "question": parsed.question,
                "steps": parsed.steps,
                "broken_step": None,
                "error_type": "unreadable",
                "knowledge_point": None,
                "confidence": 1.0,
                "explanation": None,
                "practice": [],
                "schedule": [],
                "message": RETAKE_MESSAGE,
                "flags": {
                    "unmapped": False,
                    "needs_teacher_review": True,
                    "provider": self.provider.name,
                    "elapsed_s": round(time.time() - t0, 3),
                },
            }

        raw = self.provider.diagnose(parsed.question, parsed.steps)
        self._step("diagnose", raw)
        error_type = str(raw.get("error_type", "procedural")).lower()
        if error_type not in ERROR_TYPES:
            error_type = "procedural"
        confidence = float(raw.get("confidence", 0.0) or 0.0)
        broken_step = raw.get("broken_step")

        query = " ; ".join([parsed.question] + parsed.steps)
        ground = self.grounder.ground(query, hint=str(raw.get("knowledge_point_hint", "")))
        self._step("ground", {"knowledge_point": ground.knowledge_point, "score": ground.score, "unmapped": ground.unmapped})

        node = ground.node
        items = list(node.get("items", [])) if node else []
        practice = [f"{item['id']} ({item['difficulty']})" for item in items[:3]]

        if not node:
            message = (
                f"No knowledge point in the syllabus matched this work (best score {ground.score} "
                f"< threshold {ground.threshold}). Reported as unmapped rather than guessed — "
                "please tell us the chapter or textbook."
            )
        else:
            message = None

        # Honest failure accounting: unmapped diagnoses are recorded too (grounded=0),
        # so the unmapped share of uploads is measurable instead of invisible.
        mistake_id = None
        schedule = initial_path(items)
        plan: List[Dict[str, Any]] = []
        if save:
            conn = connect(os.path.join(ROOT, self.cfg.db_path))
            init_db(conn)
            mistake_id = add_mistake(
                conn,
                {
                    "student_id": student_id,
                    "image_path": image_path,
                    "question_text": parsed.question,
                    "steps": parsed.steps,
                    "broken_step": broken_step,
                    "error_type": error_type,
                    "knowledge_point_id": node["id"] if node else None,
                    "confidence": confidence,
                    "grounded": bool(node),
                    "provider": self.provider.name,
                },
            )
            gaps = [
                Gap(
                    knowledge_point=g["knowledge_point"],
                    error_type=g["error_type"],
                    repeat_count=g["repeat_count"],
                    days_since=g["days_since"],
                    cleared_count=g.get("cleared_count", 0),
                    items=self.node_by_id(g["knowledge_point"]).get("items", []),
                )
                for g in gap_profile(conn, student_id)
                if self.node_by_id(g["knowledge_point"])
            ]
            conn.close()
            plan = build_path(gaps, max_per_day=self.cfg.max_practice_per_day)

        result = {
            "status": "ok",
            "question": parsed.question,
            "steps": parsed.steps,
            "broken_step": broken_step,
            "error_type": error_type,
            "knowledge_point": ground.knowledge_point,
            "confidence": confidence,
            "explanation": raw.get("explanation"),
            "remediation": node.get("remediation") if node else None,
            "practice": practice,
            "schedule": schedule,
            "plan": plan,
            "message": message,
            "mistake_id": mistake_id,
            "flags": {
                "unmapped": ground.unmapped,
                "needs_teacher_review": confidence < self.cfg.confidence_review,
                "provider": self.provider.name,
                "grounding_score": ground.score,
                "threshold": ground.threshold,
                "elapsed_s": round(time.time() - t0, 3),
            },
        }
        self._step("result", result)
        return result


def trace_markdown(sample_id: str, parsed: ParseResult, result: Dict[str, Any], trace: List[Dict[str, Any]]) -> str:
    lines = [
        f"# Trace · {sample_id} · {datetime.utcnow().isoformat(timespec='seconds')} UTC",
        "",
        f"- provider: `{result['flags']['provider']}`",
        f"- input status: `{parsed.status}`",
        f"- grounding: score `{result['flags'].get('grounding_score')}` vs threshold `{result['flags'].get('threshold')}` "
        f"(unmapped={result['flags']['unmapped']})",
        f"- needs teacher review: `{result['flags']['needs_teacher_review']}`",
        f"- elapsed: `{result['flags']['elapsed_s']}s`",
        "",
        "## Input",
        "",
        "```json",
        json.dumps({"question": parsed.question, "steps": parsed.steps}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Output",
        "",
        "```json",
        json.dumps(result, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Pipeline steps",
        "",
    ]
    for entry in trace:
        lines.append(f"- `{entry['step']}` @ {entry['at']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose one wrong answer.")
    ap.add_argument("--sample", help="sample JSON from samples/ (question + steps already parsed)")
    ap.add_argument("--image", help="photo of the wrong answer (needs a vision provider)")
    ap.add_argument("--question", help="question text, skips photo parsing")
    ap.add_argument("--steps", help="student's working steps, separated by ';'")
    ap.add_argument("--student", default="anon")
    ap.add_argument("--json", action="store_true", help="print raw JSON only")
    ap.add_argument("--trace", action="store_true", help="save the run to traces/")
    ap.add_argument("--offline", action="store_true", help="force the offline rule baseline")
    ap.add_argument("--no-save", action="store_true", help="do not write to SQLite")
    args = ap.parse_args(argv)

    cfg = Config.from_env()
    pipe = Pipeline(cfg, force_offline=args.offline or not cfg.configured)

    if args.sample:
        with open(args.sample, "r", encoding="utf-8") as fh:
            sample = json.load(fh)
        parsed = ParseResult(
            status=sample.get("parse_status", "ok" if sample.get("question") else "unreadable"),
            question=sample.get("question", ""),
            steps=sample.get("steps", []),
        )
        sample_id = sample.get("id", os.path.basename(args.sample))
    elif args.image:
        parsed = parse_photo(pipe.provider, args.image)
        sample_id = os.path.basename(args.image)
    elif args.question or args.steps:
        parsed = from_text(args.question or "", [s.strip() for s in (args.steps or "").split(";") if s.strip()])
        sample_id = "cli"
    else:
        ap.error("provide --sample, --image, or --question/--steps")
        return 2

    result = pipe.run(parsed, student_id=args.student, image_path=args.image, save=not args.no_save)

    if args.trace:
        os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True)
        path = os.path.join(ROOT, "traces", f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{sample_id}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(trace_markdown(sample_id, parsed, result, pipe.trace))
        result["trace_path"] = os.path.relpath(path, ROOT)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[{result['flags']['provider']}] {sample_id}")
        print(f"  status        : {result['status']}")
        print(f"  broken step   : {result['broken_step']}")
        print(f"  error type    : {result['error_type']}")
        print(f"  knowledge pt  : {result['knowledge_point']} (score {result['flags'].get('grounding_score')})")
        print(f"  confidence    : {result['confidence']}  review={result['flags']['needs_teacher_review']}")
        print(f"  practice      : {', '.join(result['practice']) or '—'}")
        print(f"  schedule      : {', '.join('day' + str(s['day']) for s in result['schedule']) or '—'}")
        if result.get("message"):
            print(f"  note          : {result['message']}")
        if result.get("trace_path"):
            print(f"  trace         : {result['trace_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
