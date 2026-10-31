# AGENTS.md — development conventions

Pathfinder AI is judged on whether a wrong answer becomes **data**, so the code is organised around one rule: nothing in the pipeline may invent.

## Non-negotiables

1. **Closed taxonomy.** `error_type` is one of `conceptual` / `procedural` / `careless` / `unreadable`. Anything else is coerced to `procedural` and logged — never silently passed through.
2. **No entry without a status.** The pipeline returns early on any parse status other than `ok`. A confident diagnosis of a misread question is the worst failure mode in this project.
3. **Grounded or unmapped.** A knowledge point below `GROUNDING_THRESHOLD` is reported `unmapped` with its score. Never fall back to "closest enough".
4. **Practice items are retrieved, never generated.** Items come from `knowledge_base/knowledge_points.json`. Adding freehand LLM-generated exercises is a review-blocking change.
5. **Mistakes are append-only.** `mistakes` rows are never updated; gap profiles are recomputed from history.
6. **Provider is always labelled.** Every result carries `flags.provider`. `offline-rule-baseline` output must never be presented as a model diagnosis.
7. **Failure modes are explicit.** Auth / quota / network / malformed-response each raise a distinct `llm.LLMError` subclass; no bare `except: pass`.

## Layout rule

Core logic lives in `diagnose.py` as a single pipeline. `app.py` (Streamlit) and the CLI in `diagnose.py::main` are thin adapters — neither may contain diagnosis logic.

## Before every commit

```bash
python -m unittest discover -s tests          # must stay green, zero API spend
python diagnose.py --sample samples/alg_distributive.json --no-save
python eval.py --set samples
```

## Adding a knowledge point

Add the node to `knowledge_base/knowledge_points.json` (title, keywords, remediation, three graded items) **and** its regex matchers to `knowledge_base/matchers.json`. A node without matchers is unreachable in offline mode.

## Adding a misconception rule

Rules live in `llm.py::MISCONCEPTIONS`. Each rule needs a `blame` policy (`matching` or `first`) and a matching fixture in `samples/` with teacher labels, so `eval.py` keeps the rule honest.
