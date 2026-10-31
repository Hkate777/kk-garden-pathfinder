# Trace · alg_distributive · 2026-10-31T14:58:53 UTC

- provider: `offline-rule-baseline`
- input status: `ok`
- grounding: score `0.8621` vs threshold `0.42` (unmapped=False)
- needs teacher review: `False`
- elapsed: `0.003s`

## Input

```json
{
  "question": "Find the value of x: 2(x+3) = 4x - 6",
  "steps": [
    "2x + 3 = 4x - 6",
    "3 = 2x - 6",
    "x = 4.5"
  ]
}
```

## Output

```json
{
  "status": "ok",
  "question": "Find the value of x: 2(x+3) = 4x - 6",
  "steps": [
    "2x + 3 = 4x - 6",
    "3 = 2x - 6",
    "x = 4.5"
  ],
  "broken_step": "2x + 3 = 4x - 6",
  "error_type": "conceptual",
  "knowledge_point": "ALG-7.3",
  "confidence": 0.9,
  "explanation": "Rule baseline matched the 'distributive_drop' pattern. A teacher should confirm.",
  "remediation": "The factor multiplies every term inside the bracket, carrying its sign: a(b+c) = ab + ac, and a(b-c) = ab - ac.",
  "practice": [
    "ALG-7.3 #1 (easy)",
    "ALG-7.3 #2 (medium)",
    "ALG-7.3 #3 (applied)"
  ],
  "schedule": [
    {
      "day": 1,
      "item_id": "ALG-7.3 #1",
      "text": "Expand 2(x+3)",
      "difficulty": "easy"
    },
    {
      "day": 3,
      "item_id": "ALG-7.3 #2",
      "text": "Expand and solve 2(x+3) = 4x - 6",
      "difficulty": "medium"
    },
    {
      "day": 7,
      "item_id": "ALG-7.3 #3",
      "text": "Expand -3(2y-5) and use it to solve -3(2y-5) = 9",
      "difficulty": "applied"
    }
  ],
  "plan": [],
  "message": null,
  "mistake_id": null,
  "flags": {
    "unmapped": false,
    "needs_teacher_review": false,
    "provider": "offline-rule-baseline",
    "grounding_score": 0.8621,
    "threshold": 0.42,
    "elapsed_s": 0.003
  }
}
```

## Pipeline steps

- `parse` @ 2026-10-31T14:58:53
- `diagnose` @ 2026-10-31T14:58:53
- `ground` @ 2026-10-31T14:58:53
- `result` @ 2026-10-31T14:58:53
