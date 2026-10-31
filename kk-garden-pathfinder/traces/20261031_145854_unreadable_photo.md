# Trace · unreadable_photo · 2026-10-31T14:58:54 UTC

- provider: `offline-rule-baseline`
- input status: `unreadable`
- grounding: score `None` vs threshold `None` (unmapped=False)
- needs teacher review: `True`
- elapsed: `0.0s`

## Input

```json
{
  "question": "",
  "steps": []
}
```

## Output

```json
{
  "status": "unreadable",
  "question": "",
  "steps": [],
  "broken_step": null,
  "error_type": "unreadable",
  "knowledge_point": null,
  "confidence": 1.0,
  "explanation": null,
  "practice": [],
  "schedule": [],
  "message": "The photo could not be read reliably. Please retake it straight-on, well lit and uncropped.",
  "flags": {
    "unmapped": false,
    "needs_teacher_review": true,
    "provider": "offline-rule-baseline",
    "elapsed_s": 0.0
  }
}
```

## Pipeline steps

- `parse` @ 2026-10-31T14:58:54
- `abort` @ 2026-10-31T14:58:54
