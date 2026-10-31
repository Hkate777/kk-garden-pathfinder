# Trace · phy_unit_conversion · 2026-10-31T14:58:53 UTC

- provider: `offline-rule-baseline`
- input status: `ok`
- grounding: score `0.5856` vs threshold `0.42` (unmapped=False)
- needs teacher review: `False`
- elapsed: `0.004s`

## Input

```json
{
  "question": "A 500 g mass is pushed with a net force of 10 N. Find its acceleration.",
  "steps": [
    "a = F/m = 10/500",
    "a = 0.02 m/s^2"
  ]
}
```

## Output

```json
{
  "status": "ok",
  "question": "A 500 g mass is pushed with a net force of 10 N. Find its acceleration.",
  "steps": [
    "a = F/m = 10/500",
    "a = 0.02 m/s^2"
  ],
  "broken_step": "a = F/m = 10/500",
  "error_type": "procedural",
  "knowledge_point": "PHY-9.2",
  "confidence": 0.85,
  "explanation": "Rule baseline matched the 'unit_not_converted' pattern. A teacher should confirm.",
  "remediation": "Convert every quantity to SI before substituting: grams to kilograms, km/h to m/s (divide by 3.6).",
  "practice": [
    "PHY-9.2 #1 (easy)",
    "PHY-9.2 #2 (medium)",
    "PHY-9.2 #3 (applied)"
  ],
  "schedule": [
    {
      "day": 1,
      "item_id": "PHY-9.2 #1",
      "text": "Convert 500 g to kg",
      "difficulty": "easy"
    },
    {
      "day": 3,
      "item_id": "PHY-9.2 #2",
      "text": "Convert 72 km/h to m/s",
      "difficulty": "medium"
    },
    {
      "day": 7,
      "item_id": "PHY-9.2 #3",
      "text": "A 500 g mass is pushed with 10 N. Find its acceleration in SI units.",
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
    "grounding_score": 0.5856,
    "threshold": 0.42,
    "elapsed_s": 0.004
  }
}
```

## Pipeline steps

- `parse` @ 2026-10-31T14:58:53
- `diagnose` @ 2026-10-31T14:58:53
- `ground` @ 2026-10-31T14:58:53
- `result` @ 2026-10-31T14:58:53
