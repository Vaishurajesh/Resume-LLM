# Part 4 — Evaluation Framework for the Resume Intelligence Model

## 1. What Metrics to Use

A resume-parsing model needs metrics at three levels: **structural validity**,
**field-level correctness**, and **behavioral safety** (hallucination). No
single metric captures all three, so the framework tracks:

| Level | Metric | What it catches |
|---|---|---|
| Structural | JSON/schema validity rate | Malformed output, missing required keys, wrong types |
| Field-level | Precision, Recall, F1 per field (scalar fields); set-F1 for lists (skills, certifications); sequence-alignment F1 for nested lists (experience, education) | Wrong or missing extracted values |
| Field-level | Exact-match accuracy for scalar fields (name, email, phone) | Strict correctness where fuzzy matching doesn't apply |
| Behavioral | Hallucination rate | Model inventing data not present in input |
| Behavioral | Null-fidelity rate | Correctly outputting `null`/`[]` when a field is genuinely absent, instead of guessing |
| Aggregate | Exact full-record match rate | Strictest metric — whole JSON object matches ground truth exactly |

**Why this mix:** JSON validity alone would let a model "cheat" by
returning `{}` for every input (100% valid, 0% useful). Field-level F1
alone wouldn't catch a model that invents plausible-looking values.
Combining them prevents either failure mode from going unnoticed.

## 2. Measuring Parsing Accuracy

For each test example, compare predicted JSON to ground-truth JSON field by field:

- **Scalar fields** (`name`, `email`, `phone`, `location`, `summary`):
  normalize (lowercase, strip whitespace/punctuation) then exact-match.
  Report accuracy = matches / total non-null ground-truth fields.
- **List-of-strings fields** (`skills`, `certifications`): treat as sets.
  Precision = |predicted ∩ truth| / |predicted|, Recall = |predicted ∩
  truth| / |truth|, F1 = harmonic mean. Use fuzzy string matching (e.g.
  Levenshtein ratio > 0.85) so "Python" vs "python programming" still counts.
- **List-of-objects fields** (`experience`, `education`): align predicted
  and ground-truth entries via the Hungarian algorithm (or greedy
  best-match) on a similarity score (weighted match of title/degree +
  company/institution + dates), then compute per-field F1 within matched
  pairs, and count unmatched entries as full misses.
- **Aggregate score:** weighted average across fields, with experience
  and education weighted higher since they carry the most business value
  in a resume platform.

## 3. Detecting Hallucinations

Hallucination = the model outputs a value that does not appear (verbatim
or as a close paraphrase) anywhere in the source input text.

Method:
1. For every non-null string value in the predicted output (name, company
   names, skills, degree names, etc.), check if it (or a normalized/fuzzy
   variant) appears in the input text.
2. Flag any value that doesn't appear as a **potential hallucination**.
3. Distinguish **benign normalization** (e.g., model writes "2021-06" when
   input said "June 2021" — not a hallucination, just reformatting) from
   **fabrication** (e.g., model invents a company name never mentioned)
   using a fuzzy/semantic similarity threshold rather than exact string match.
4. Report **hallucination rate** = flagged-fabricated-values / total
   extracted values, tracked per field so we know if hallucination
   clusters in a specific field (e.g. `summary`, which is more
   generative and thus more hallucination-prone than `email`).

This is the single most important safety metric for this use case: a
resume platform that invents a candidate's skills or experience is a
serious trust and legal problem, worse than one that simply misses a field.

## 4. Validating JSON Correctness

Three layers, from cheapest to most expensive:

1. **Syntactic validity:** `json.loads()` succeeds without exception.
2. **Schema validity:** parsed object matches a `jsonschema` definition —
   correct top-level keys, correct types (`skills` is a list, not a
   string; `experience` entries have the required sub-keys), no
   unexpected extra top-level keys.
3. **Semantic validity:** dates parse as valid dates or recognized
   relative terms ("Present"), email fields match an email regex if
   non-null, list fields aren't padded with empty/duplicate entries.

Any output failing layer 1 is scored as a complete failure for that
example (0 across all field-level metrics) rather than attempting partial
credit, since a downstream system can't consume invalid JSON at all.

## 5. Comparing Two Model Versions

- Run both model versions over the **same held-out test set** (not
  training data) and compute the full metric suite for each.
- Report a **side-by-side scorecard** (validity rate, per-field F1,
  hallucination rate, full-match rate) rather than a single number, since
  a new version might trade off differently (e.g. higher recall but more
  hallucination).
- Run a **paired significance test** (e.g. McNemar's test on per-example
  pass/fail, or a paired bootstrap over F1 scores) rather than just
  comparing means, since test sets are usually small (tens to low
  hundreds of examples) and raw differences can be noise.
- Maintain a **fixed "hard case" subset** (garbled multi-column, OCR
  noise, missing-field examples) and report scores on that subset
  separately — aggregate improvement can hide regression on exactly the
  edge cases that matter most in production.

## 6. Regression Testing After Retraining

- Maintain a **frozen regression test set** that is never used for
  training, covering every edge-case category in the dataset (missing
  fields, garbled text, non-English, gaps, etc.) plus real anonymized
  production failures logged over time.
- After every retrain, run the new checkpoint against this fixed set and
  require it to **not regress** below the previous checkpoint's score on
  any individual category by more than a small tolerance (e.g. 2
  percentage points) — this catches "fixed the new bug, broke an old one"
  cycles.
- Gate deployment on regression-suite pass, not just aggregate-score
  improvement: a model can improve overall average while quietly breaking
  a previously-working category, which the regression suite is
  specifically designed to catch.
- Version and store every regression run's full output (not just the
  score) so failures can be diffed against the prior version's output to
  see exactly what changed.

## 7. Automating the Evaluation (Bonus)

The accompanying `evaluate.py` script automates:
- Batched inference over a test set given a model + (optional) LoRA adapter
- JSON/schema validation (layers 1–2 above)
- Field-level precision/recall/F1 computation with fuzzy matching for lists
- Hallucination flagging via substring/fuzzy-match against source input
- A single aggregate report (JSON + printed summary table) suitable for
  CI-style automated regression gating — exit code is non-zero if any
  metric drops below a configurable threshold, so it can be wired into a
  GitHub Actions workflow that blocks merging a new fine-tune until it
  passes the regression bar.
