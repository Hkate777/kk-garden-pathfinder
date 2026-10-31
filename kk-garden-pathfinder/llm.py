"""Pluggable inference layer.

Two providers implement the same interface:
  * OpenAICompatProvider  — any OpenAI-compatible endpoint (chat + vision + embeddings)
  * StubProvider          — offline rule baseline, used when no API key is configured

The stub is NOT a model. It is a deterministic rule engine that lets the whole
pipeline run, be tested and be demoed without network, and every output it produces
is labelled `provider: offline-rule-baseline` so it can never be mistaken for a
model diagnosis.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests

ERROR_TYPES = ("conceptual", "procedural", "careless", "unreadable")

DIAGNOSE_SYSTEM = (
    "You are a mathematics and science diagnosis engine for school-level work. "
    "Given a question and the student's own working steps, locate the FIRST step where the "
    "reasoning broke, classify the error, and explain it in one or two sentences for a student. "
    "Reply with JSON only, matching this schema:\n"
    '{"broken_step": str, "error_type": "conceptual"|"procedural"|"careless"|"unreadable", '
    '"knowledge_point_hint": str, "confidence": float, "explanation": str}\n'
    "error_type is a closed set: conceptual = wrong idea, procedural = right idea executed wrongly, "
    "careless = arithmetic or sign slip, unreadable = the work cannot be read. "
    "Never invent a step the student did not write. If the work cannot be read, say so."
)

VISION_SYSTEM = (
    "You transcribe a photo of a school exercise. Reply with JSON only: "
    '{"status": "ok"|"ambiguous"|"unreadable"|"rotated", "question": str, "steps": [str], "notes": str}. '
    "Keep the student's own notation and mistakes exactly as written. "
    "If the image is blurry, cut off or rotated, set status accordingly — never guess missing text."
)


class LLMError(Exception):
    """Base class. Every failure mode below is actionable, never silent."""


class LLMNotConfiguredError(LLMError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMQuotaError(LLMError):
    pass


class LLMNetworkError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class VisionUnavailableError(LLMError):
    pass


@dataclass
class Config:
    base_url: str = ""
    api_key: str = ""
    vision_model: str = ""
    text_model: str = ""
    embedding_model: str = ""
    timeout: int = 60
    max_retries: int = 3
    offline: bool = False
    grounding_threshold: float = 0.42
    confidence_review: float = 0.60
    max_practice_per_day: int = 3
    db_path: str = "data/pathfinder.db"
    kb_path: str = "knowledge_base/knowledge_points.json"
    matchers_path: str = "knowledge_base/matchers.json"

    @classmethod
    def from_env(cls) -> "Config":
        def f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except ValueError:
                return default

        offline = os.getenv("PATHFINDER_OFFLINE", "0").strip().lower() in ("1", "true", "yes")
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "").strip(),
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            vision_model=os.getenv("VISION_MODEL", "").strip(),
            text_model=os.getenv("TEXT_MODEL", "").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
            grounding_threshold=f("GROUNDING_THRESHOLD", 0.42),
            confidence_review=f("CONFIDENCE_REVIEW", 0.60),
            max_practice_per_day=int(f("MAX_PRACTICE_PER_DAY", 3)),
            db_path=os.getenv("DB_PATH", "data/pathfinder.db").strip(),
            kb_path=os.getenv("KB_PATH", "knowledge_base/knowledge_points.json").strip(),
            matchers_path=os.getenv("MATCHERS_PATH", "knowledge_base/matchers.json").strip(),
            offline=offline,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.text_model)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _tolerant_json(text: str) -> Dict[str, Any]:
    """Accept raw JSON, fenced JSON, or JSON embedded in prose."""
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMResponseError(f"model did not return parseable JSON: {text[:200]!r}")


# --------------------------------------------------------------------------
# OpenAI-compatible provider
# --------------------------------------------------------------------------
class OpenAICompatProvider:
    name = "openai-compatible"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url.rstrip('/')}{path}"
        last: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.cfg.timeout,
                )
            except requests.exceptions.RequestException as exc:  # network layer
                last = exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (401, 403):
                raise LLMAuthError(f"{resp.status_code}: check LLM_API_KEY")
            if resp.status_code == 429:
                if attempt == self.cfg.max_retries - 1:
                    raise LLMQuotaError("429: quota exhausted — check your plan or wait")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 500:
                last = LLMNetworkError(f"{resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise LLMResponseError(f"{resp.status_code}: {resp.text[:200]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMResponseError(f"malformed JSON body: {resp.text[:200]}") from exc
        raise LLMNetworkError(f"request failed after {self.cfg.max_retries} attempts: {last}")

    def chat(self, system: str, user: str) -> str:
        body = self._post(
            "/chat/completions",
            {
                "model": self.cfg.text_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
        )
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMResponseError(f"unexpected chat response shape: {str(body)[:200]}") from exc

    def parse_photo(self, image_path: str) -> Dict[str, Any]:
        if not self.cfg.vision_model:
            raise VisionUnavailableError("VISION_MODEL is not set — photo parsing unavailable")
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        body = self._post(
            "/chat/completions",
            {
                "model": self.cfg.vision_model,
                "messages": [
                    {"role": "system", "content": VISION_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe this exercise exactly as written."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    },
                ],
                "temperature": 0,
            },
        )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMResponseError(f"unexpected vision response shape: {str(body)[:200]}") from exc
        return _tolerant_json(content)

    def diagnose(self, question: str, steps: List[str]) -> Dict[str, Any]:
        payload = json.dumps({"question": question, "steps": steps}, ensure_ascii=False)
        return _tolerant_json(self.chat(DIAGNOSE_SYSTEM, payload))

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not self.cfg.embedding_model:
            raise LLMNotConfiguredError("EMBEDDING_MODEL is not set")
        body = self._post("/embeddings", {"model": self.cfg.embedding_model, "input": texts})
        try:
            return [row["embedding"] for row in body["data"]]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError(f"unexpected embeddings response shape: {str(body)[:200]}") from exc


# --------------------------------------------------------------------------
# Offline rule baseline
# --------------------------------------------------------------------------
@dataclass
class Misconception:
    name: str
    pattern: str
    error_type: str
    knowledge_point: str
    confidence: float = 0.9
    blame: str = "matching"  # matching → the step that matches; first → the student's first transformation

    def search(self, text: str):
        return re.search(self.pattern, text, flags=re.IGNORECASE)


MISCONCEPTIONS: List[Misconception] = [
    Misconception("distributive_drop", r"\d+\s*\(\s*[a-z]\s*[+-]\s*\d+\s*\)\s*=\s*\d*[a-z]\s*[+-]\s*\d+", "conceptual", "ALG-7.3"),
    Misconception("square_of_sum", r"\(\s*[a-z]\s*[+-]\s*[a-z]\s*\)\s*\^?\s*2\s*=\s*[a-z]\s*\^?\s*2\s*[+]\s*[a-z]\s*\^?\s*2", "conceptual", "ALG-8.3"),
    Misconception("fraction_add_across", r"\d+\s*/\s*\d+\s*[+]\s*\d+\s*/\s*\d+\s*=\s*\d+\s*/\s*\d+", "conceptual", "ALG-7.4"),
    Misconception("sign_drop_transpose", r"([a-z])\s*-\s*(\d+)\s*=\s*(\d+)[\s\S]{0,80}?\1\s*=\s*\3\s*-\s*\2", "careless", "ALG-7.2", 0.85, "first"),
    Misconception("unit_not_converted", r"\d+\s*g\b[\s\S]{0,200}?/\s*\d+", "procedural", "PHY-9.2", 0.85, "first"),
    Misconception("unbalanced_equation", r"[A-Z][a-z]?\d*\s*\+\s*[A-Z][a-z]?\d*\s*->\s*[A-Z]", "procedural", "CHE-8.1"),
]


class StubProvider:
    """Deterministic rule baseline. No network, no model — labelled in every output."""

    name = "offline-rule-baseline"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def parse_photo(self, image_path: str) -> Dict[str, Any]:
        raise VisionUnavailableError(
            "offline mode has no vision provider — pass --question/--steps or use a sample from samples/"
        )

    def diagnose(self, question: str, steps: List[str]) -> Dict[str, Any]:
        if not question and not steps:
            return {
                "broken_step": None,
                "error_type": "unreadable",
                "knowledge_point_hint": "",
                "confidence": 1.0,
                "explanation": "The work could not be read. Please upload a clearer, straight-on photo.",
            }
        joined = " ; ".join([question] + list(steps))
        for rule in MISCONCEPTIONS:
            hit = rule.search(joined)
            if hit:
                broken = next((s for s in steps if re.search(rule.pattern, s, flags=re.IGNORECASE)), None)
                if broken is None:
                    if rule.blame == "first" and steps:
                        broken = steps[0]
                    elif rule.search(question) and steps:
                        broken = steps[0]
                    else:
                        broken = steps[-1] if steps else None
                return {
                    "broken_step": broken,
                    "error_type": rule.error_type,
                    "knowledge_point_hint": rule.knowledge_point,
                    "confidence": rule.confidence,
                    "explanation": f"Rule baseline matched the '{rule.name}' pattern. A teacher should confirm.",
                }
        return {
            "broken_step": steps[-1] if steps else None,
            "error_type": "procedural",
            "knowledge_point_hint": "",
            "confidence": 0.35,
            "explanation": "No known misconception pattern matched; the last written step is flagged for a teacher's review.",
        }

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Hashed bag-of-words vectors — deterministic, no network, unit-normalised."""
        dim = 256
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * dim
            for token in re.findall(r"[a-z0-9]+", str(text).lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_provider(cfg: Optional[Config] = None, force_offline: bool = False) -> Any:
    cfg = cfg or Config.from_env()
    if force_offline or cfg.offline or not cfg.configured:
        return StubProvider(cfg)
    return OpenAICompatProvider(cfg)
