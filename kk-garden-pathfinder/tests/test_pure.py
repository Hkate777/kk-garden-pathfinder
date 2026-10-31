"""Offline unit tests — no API calls, no API spend.

Run:  python -m unittest discover -s tests -v      (or: python tests/test_pure.py)
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diagnose import Pipeline  # noqa: E402
from grounding import Grounder, load_kb  # noqa: E402
from llm import ERROR_TYPES, Config, StubProvider, _tolerant_json  # noqa: E402
from schedule import Gap, SEVERITY, build_path, initial_path, rank_gaps, risk  # noqa: E402
from store import (  # noqa: E402
    add_mistake,
    class_heatmap,
    connect,
    gap_profile,
    init_db,
    record_practice,
    repeat_error_rate,
)
from vision import RETAKE_MESSAGE, ParseResult, from_text, parse_photo  # noqa: E402


def kb_paths():
    return (
        os.path.join(ROOT, "knowledge_base", "knowledge_points.json"),
        os.path.join(ROOT, "knowledge_base", "matchers.json"),
    )


class TestTaxonomy(unittest.TestCase):
    def test_closed_set(self):
        self.assertEqual(set(ERROR_TYPES), {"conceptual", "procedural", "careless", "unreadable"})

    def test_severity_covers_every_type(self):
        for t in ERROR_TYPES:
            self.assertIn(t, SEVERITY)


class TestJSONTolerance(unittest.TestCase):
    def test_fenced_json(self):
        self.assertEqual(_tolerant_json("```json\n{\"a\": 1}\n```"), {"a": 1})

    def test_json_in_prose(self):
        self.assertEqual(_tolerant_json('sure: {"broken_step": "x = 6"} done'), {"broken_step": "x = 6"})


class TestGrounding(unittest.TestCase):
    def setUp(self):
        nodes, matchers = load_kb(*kb_paths())
        self.nodes, self.matchers = nodes, matchers
        self.grounder = Grounder(nodes, matchers, StubProvider(Config()))

    def test_maps_distributive_error(self):
        res = self.grounder.ground("2(x+3) = 4x - 6 ; 2x + 3 = 4x - 6", hint="distributive property")
        self.assertEqual(res.knowledge_point, "ALG-7.3")
        self.assertFalse(res.unmapped)

    def test_unmapped_below_threshold(self):
        strict = Grounder(self.nodes, self.matchers, StubProvider(Config()), threshold=0.99)
        res = strict.ground("2(x+3) = 4x - 6")
        self.assertTrue(res.unmapped)
        self.assertIsNone(res.node)

    def test_unrelated_text_is_unmapped(self):
        res = self.grounder.ground("zzz qqq random unrelated sentence")
        self.assertTrue(res.unmapped)


class TestSchedule(unittest.TestCase):
    def test_initial_path_is_spaced(self):
        items = [{"id": "a", "difficulty": "easy", "text": "A"}, {"id": "b", "difficulty": "medium", "text": "B"},
                 {"id": "c", "difficulty": "applied", "text": "C"}]
        path = initial_path(items)
        self.assertEqual([p["day"] for p in path], [1, 3, 7])

    def test_daily_cap(self):
        items = [{"id": f"i{n}", "difficulty": "easy", "text": str(n)} for n in range(9)]
        gaps = [Gap("KP-A", "conceptual", 3, 1, items=items),
                Gap("KP-B", "conceptual", 3, 1, items=items),
                Gap("KP-C", "careless", 1, 1, items=items)]
        for block in build_path(gaps, max_per_day=3):
            self.assertLessEqual(len(block["items"]), 3)

    def test_conceptual_outranks_careless_at_equal_repeat(self):
        gaps = [Gap("KP-C", "careless", 1, 0), Gap("KP-A", "conceptual", 1, 0)]
        ranked = rank_gaps(gaps)
        self.assertEqual(ranked[0].knowledge_point, "KP-A")

    def test_risk_decays_with_clears(self):
        self.assertLess(risk(2, 0, "conceptual", cleared_count=2), risk(2, 0, "conceptual", cleared_count=0))

    def test_recency_weight_decays(self):
        self.assertGreater(risk(1, 0, "conceptual"), risk(1, 30, "conceptual"))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.conn = connect(os.path.join(self.tmp, "test.db"))
        init_db(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_profile_uses_latest_error_type(self):
        add_mistake(self.conn, {"student_id": "s9", "created_at": "2026-10-01T00:00:00", "question_text": "q1",
                                "steps": ["a"], "error_type": "conceptual", "knowledge_point_id": "ALG-7.3",
                                "confidence": 0.9, "grounded": True, "provider": "test"})
        add_mistake(self.conn, {"student_id": "s9", "question_text": "q2", "steps": ["b"],
                                "error_type": "careless", "knowledge_point_id": "ALG-7.3",
                                "confidence": 0.5, "grounded": True, "provider": "test"})
        self.assertEqual(gap_profile(self.conn, "s9")[0]["error_type"], "careless")

    def test_mistake_roundtrip_and_profile(self):
        add_mistake(self.conn, {"student_id": "s1", "question_text": "2(x+3)=4x-6", "steps": ["2x+3=4x-6"],
                                "error_type": "conceptual", "knowledge_point_id": "ALG-7.3", "confidence": 0.9,
                                "grounded": True, "provider": "test"})
        add_mistake(self.conn, {"student_id": "s1", "question_text": "3(x+1)=9", "steps": ["3x+1=9"],
                                "error_type": "conceptual", "knowledge_point_id": "ALG-7.3", "confidence": 0.8,
                                "grounded": True, "provider": "test"})
        profile = gap_profile(self.conn, "s1")
        self.assertEqual(profile[0]["knowledge_point"], "ALG-7.3")
        self.assertEqual(profile[0]["repeat_count"], 2)

    def test_repeat_error_rate_counts_repeats(self):
        for i in range(3):
            add_mistake(self.conn, {"student_id": "s2", "question_text": f"q{i}", "steps": ["a"],
                                    "error_type": "careless", "knowledge_point_id": "ALG-7.2", "confidence": 0.5,
                                    "grounded": True, "provider": "test"})
        add_mistake(self.conn, {"student_id": "s2", "question_text": "q-other", "steps": ["b"],
                                "error_type": "careless", "knowledge_point_id": "ALG-7.4", "confidence": 0.5,
                                "grounded": True, "provider": "test"})
        stats = repeat_error_rate(self.conn, "s2")
        self.assertEqual(stats["repeat_knowledge_points"], 1)
        self.assertAlmostEqual(stats["repeat_error_rate"], 0.5)

    def test_practice_log_records_mastery(self):
        mid = add_mistake(self.conn, {"student_id": "s3", "question_text": "q", "steps": ["a"],
                                      "error_type": "procedural", "knowledge_point_id": "ALG-7.2",
                                      "confidence": 0.7, "grounded": True, "provider": "test"})
        record_practice(self.conn, mid, "ALG-7.2 #1", 1, "pass")
        self.assertEqual(gap_profile(self.conn, "s3")[0]["cleared_count"], 1)

    def test_class_heatmap(self):
        add_mistake(self.conn, {"student_id": "s4", "question_text": "q", "steps": ["a"],
                                "error_type": "careless", "knowledge_point_id": "PHY-9.2", "confidence": 0.6,
                                "grounded": True, "provider": "test"})
        self.assertEqual(class_heatmap(self.conn)[0]["knowledge_point"], "PHY-9.2")

    def test_heatmap_excludes_unmapped_mistakes(self):
        add_mistake(self.conn, {"student_id": "s5", "question_text": "q", "steps": ["a"],
                                "error_type": "careless", "knowledge_point_id": None, "confidence": 0.4,
                                "grounded": False, "provider": "test"})
        self.assertEqual(class_heatmap(self.conn), [])


class TestPipelineOffline(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline(force_offline=True)

    def _run_sample(self, name):
        import json
        with open(os.path.join(ROOT, "samples", name), "r", encoding="utf-8") as fh:
            s = json.load(fh)
        parsed = ParseResult(status=s.get("parse_status", "ok" if s.get("question") else "unreadable"),
                             question=s.get("question", ""), steps=s.get("steps", []))
        return s, self.pipe.run(parsed, student_id="test", save=False)

    def test_distributive_sample_end_to_end(self):
        sample, res = self._run_sample("alg_distributive.json")
        self.assertEqual(res["error_type"], sample["expected"]["error_type"])
        self.assertEqual(res["knowledge_point"], sample["expected"]["knowledge_point"])
        self.assertIn(sample["expected"]["broken_step_contains"], res["broken_step"])
        self.assertEqual([p["day"] for p in res["schedule"]], [1, 3, 7])
        self.assertEqual(len(res["practice"]), 3)

    def test_unreadable_routes_to_retake(self):
        _, res = self._run_sample("unreadable_photo.json")
        self.assertEqual(res["error_type"], "unreadable")
        self.assertIsNone(res["knowledge_point"])
        self.assertEqual(res["schedule"], [])
        self.assertIn("retake", res["message"].lower())

    def test_unmapped_diagnosis_is_recorded_for_accounting(self):
        tmp = tempfile.mkdtemp()
        os.environ["DB_PATH"] = os.path.join(tmp, "acct.db")
        try:
            pipe = Pipeline(force_offline=True)
            parsed = from_text("zzz qqq unrelated sentence", ["some step"])
            res = pipe.run(parsed, student_id="acct", save=True)
            self.assertTrue(res["flags"]["unmapped"])
            conn = connect(os.environ["DB_PATH"])
            row = conn.execute("SELECT grounded, knowledge_point_id FROM mistakes").fetchone()
            conn.close()
            self.assertEqual(row["grounded"], 0)
            self.assertIsNone(row["knowledge_point_id"])
        finally:
            os.environ.pop("DB_PATH", None)

    def test_offline_photo_raises_explicit_status(self):
        parsed = parse_photo(self.pipe.provider, "does-not-exist.jpg")
        self.assertEqual(parsed.status, "unreadable")

    def test_empty_text_is_unreadable(self):
        self.assertEqual(from_text("", []).status, "unreadable")
        self.assertEqual(from_text("", []).notes, RETAKE_MESSAGE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
