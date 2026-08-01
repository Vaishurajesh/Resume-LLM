# Resume Intelligence LLM — Domain-Specific Fine-Tuning Prototype

A prototype demonstrating how to replace a hosted GPT-4 Mini-style API
with a fine-tuned, self-hosted open-source LLM for structured resume
parsing — covering model selection, dataset creation, LoRA fine-tuning,
evaluation methodology, and a minimal inference service.

## Project Structure

```
resume-intelligence-llm/
├── README.md                          ← you are here
├── docs/
│   └── research_document.md           ← Part 1: model selection & reasoning
├── dataset/
│   ├── resume_instructions.jsonl      ← Part 2: 50-example instruction-tuning dataset
│   ├── generate_dataset.py            ← script that builds the dataset
│   └── README.md                      ← dataset notes + scaling strategy
├── training/
│   ├── Resume_LLM_FineTuning.ipynb    ← Part 3: LoRA fine-tuning notebook (Colab, T4 GPU)
│          ← trained LoRA adapter weights
├── evaluation/
│   ├── eval_framework.md              ← Part 4: evaluation methodology
│   ├── evaluate.py                    ← automated evaluation script              ← sample evaluation output (optional)
└── inference/
    ├── config.py                      ← Part 5: configuration management
    ├── resume_parser.py                ← core inference engine
    ├── app.py                         ← FastAPI REST API
    ├── cli.py                         ← command-line interface
    

## Summary

**Task:** Given raw resume text (from PDF extraction, pasted text, or
OCR), extract structured JSON — name, contact info, education, work
experience, skills, and certifications — reliably across diverse resume
formats, missing fields, and messy real-world extraction noise.

**Model chosen:** [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(Apache 2.0 license) — small enough for cheap inference and fast LoRA
fine-tuning cycles, with strong out-of-the-box JSON/structured-output
behavior. Full reasoning in `docs/research_document.md`.

**Fine-tuning approach:** LoRA (rank 16, alpha 32) on all attention +
MLP projection layers, trained on a 50-example hand-curated instruction
dataset covering clean resumes, missing fields, garbled multi-column PDF
extraction, non-English resumes, OCR noise, and other edge cases. Trained
on a free-tier Google Colab T4 GPU. Full details in `training/`.

**Evaluation approach:** JSON/schema validity rate, field-level
precision/recall/F1 (with fuzzy matching for lists and nested
experience/education entries), a rule-based hallucination detector, and
a regression-testing strategy for comparing model versions after
retraining. Full methodology and an automated script in `evaluation/`.

**Engineering:** A minimal FastAPI service (`inference/app.py`) plus a
CLI (`inference/cli.py`), both built on a shared `ResumeParser` engine
with centralized configuration, explicit error handling (distinct HTTP
status codes for load failures, bad input, and unparseable model
output), and schema-guaranteed output shape.

## Quick Start

```bash
# 1. Fine-tune (optional — a trained adapter is already included)
# Open training/Resume_LLM_FineTuning.ipynb in Google Colab, Runtime > Run all

# 2. Evaluate
cd evaluation
python evaluate.py --test_file ../dataset/resume_instructions.jsonl \
                    --adapter_path ../training/resume-lora-adapter

# 3. Run inference
cd ../inference
pip install -r requirements.txt
python cli.py --text "Jane Doe, jane@mail.com, Product Manager at Acme (2020-Present)"
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

## Key Design Decisions & Honest Notes

- **Model size vs. capability tradeoff:** chose a 1.5B model over larger
  alternatives (Llama-3.2-3B, Mistral-7B) because resume field extraction
  is a narrow structured-extraction task that doesn't need large-model
  reasoning capacity — see `docs/research_document.md` for the full
  comparison and licensing rationale.
- **Dataset prioritizes diversity over volume:** 50 examples were chosen
  to maximize coverage of distinct failure modes (missing fields, OCR
  noise, multi-column garbling, non-English text, ambiguous dates) rather
  than many near-duplicate "clean" examples — see `dataset/README.md` for
  the scaling strategy to thousands of examples.
- **Training format caveat:** the notebook formats target outputs using
  Python's default string representation of the label dict rather than
  `json.dumps()`. This means strict `json.loads()` validation on raw
  model output will under-report validity; the evaluation script handles
  this with an `ast.literal_eval` fallback, and a production version
  should fix the training-data formatting to emit proper JSON syntax
  directly.
- **Adapter fallback behavior:** the inference engine logs a warning and
  falls back to the un-fine-tuned base model if the adapter path is
  invalid, rather than failing hard — reasonable for a prototype/demo,
  but flagged in `inference/README.md` as something to reconsider for a
  real deployment.

## License

This is an educational/assessment prototype. The base model
(`Qwen2.5-1.5B-Instruct`) is Apache 2.0 licensed.
