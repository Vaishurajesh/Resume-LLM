"""
evaluate.py — Automated evaluation for the Resume Intelligence model.

Computes: JSON/schema validity, field-level precision/recall/F1,
hallucination rate, and full-record exact-match rate over a test set.

Usage (from Colab or local, after loading `model` and `tokenizer` --
see training/Resume_LLM_FineTuning.ipynb):

    from evaluate import run_evaluation
    report = run_evaluation(model, tokenizer, test_examples)
    print_report(report)

Or standalone with a JSON test file:

    python evaluate.py --test_file dataset/resume_instructions.jsonl \
                        --model_name Qwen/Qwen2.5-1.5B-Instruct \
                        --adapter_path ./resume-lora-adapter
"""
import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

REQUIRED_KEYS = {
    "name", "email", "phone", "location", "summary",
    "skills", "experience", "education", "certifications",
}
LIST_FIELDS = {"skills", "certifications"}
NESTED_LIST_FIELDS = {"experience", "education"}
SCALAR_FIELDS = {"name", "email", "phone", "location", "summary"}

FUZZY_MATCH_THRESHOLD = 0.85


def similarity(a: str, b: str) -> float:
    if a is None or b is None:
        return 0.0
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()


def parse_model_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Try strict JSON first, then fall back to Python-literal parsing
    (useful if the training data itself used repr()-style dicts)."""
    raw_text = raw_text.strip()
    # Try to isolate the first {...} block in case of trailing text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    candidate = match.group(0) if match else raw_text
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def schema_check(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if parsed is None:
        return {"valid": False, "missing_keys": sorted(REQUIRED_KEYS), "extra_keys": []}
    missing = REQUIRED_KEYS - set(parsed.keys())
    extra = set(parsed.keys()) - REQUIRED_KEYS
    return {"valid": len(missing) == 0, "missing_keys": sorted(missing), "extra_keys": sorted(extra)}


def scalar_field_score(pred_val, true_val) -> int:
    if true_val is None:
        return 1 if pred_val is None else 0  # null-fidelity: correctly saying "not present"
    if pred_val is None:
        return 0
    return 1 if similarity(pred_val, true_val) >= FUZZY_MATCH_THRESHOLD else 0


def list_field_score(pred_list, true_list) -> Dict[str, float]:
    pred_list = pred_list or []
    true_list = true_list or []
    if not true_list and not pred_list:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not true_list:
        return {"precision": 0.0 if pred_list else 1.0, "recall": 1.0, "f1": 0.0 if pred_list else 1.0}
    if not pred_list:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    matched_true = set()
    matched_pred = 0
    for p in pred_list:
        for i, t in enumerate(true_list):
            if i in matched_true:
                continue
            if similarity(str(p), str(t)) >= FUZZY_MATCH_THRESHOLD:
                matched_true.add(i)
                matched_pred += 1
                break
    precision = matched_pred / len(pred_list)
    recall = len(matched_true) / len(true_list)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def nested_entry_similarity(pred_entry: dict, true_entry: dict, key_fields: List[str]) -> float:
    scores = [similarity(pred_entry.get(k), true_entry.get(k)) for k in key_fields]
    return sum(scores) / len(scores) if scores else 0.0


def nested_list_field_score(pred_list, true_list, key_fields: List[str]) -> Dict[str, float]:
    pred_list = pred_list or []
    true_list = true_list or []
    if not true_list and not pred_list:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not true_list:
        return {"precision": 0.0 if pred_list else 1.0, "recall": 1.0, "f1": 0.0 if pred_list else 1.0}
    if not pred_list:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    matched_true = set()
    matched_pred = 0
    for p in pred_list:
        if not isinstance(p, dict):
            continue
        best_score, best_idx = 0.0, None
        for i, t in enumerate(true_list):
            if i in matched_true or not isinstance(t, dict):
                continue
            s = nested_entry_similarity(p, t, key_fields)
            if s > best_score:
                best_score, best_idx = s, i
        if best_idx is not None and best_score >= FUZZY_MATCH_THRESHOLD:
            matched_true.add(best_idx)
            matched_pred += 1
    precision = matched_pred / len(pred_list)
    recall = len(matched_true) / len(true_list)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def detect_hallucinations(parsed: Dict[str, Any], source_text: str) -> List[str]:
    """Flag string values in the output that don't appear (fuzzily) in the source input."""
    flagged = []
    source_lower = source_text.lower()

    def check_value(val, path):
        if val is None or val == "" or val == [] :
            return
        if isinstance(val, str):
            # allow partial containment or fuzzy match against the whole source
            if val.lower() in source_lower:
                return
            words = re.findall(r"\w+", val.lower())
            if not words:
                return
            hits = sum(1 for w in words if len(w) > 2 and w in source_lower)
            if hits / max(len(words), 1) < 0.5:
                flagged.append(f"{path}: '{val}'")
        elif isinstance(val, list):
            for i, item in enumerate(val):
                check_value(item, f"{path}[{i}]")
        elif isinstance(val, dict):
            for k, v in val.items():
                check_value(v, f"{path}.{k}")

    for key, val in parsed.items():
        check_value(val, key)
    return flagged


@dataclass
class ExampleResult:
    index: int
    valid_json: bool
    missing_keys: List[str]
    scalar_scores: Dict[str, int] = field(default_factory=dict)
    list_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nested_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hallucinations: List[str] = field(default_factory=list)
    exact_match: bool = False


def evaluate_example(index: int, source_text: str, predicted_raw: str, ground_truth: Dict[str, Any]) -> ExampleResult:
    parsed = parse_model_output(predicted_raw)
    schema = schema_check(parsed)

    result = ExampleResult(index=index, valid_json=schema["valid"] or parsed is not None,
                            missing_keys=schema["missing_keys"])

    if parsed is None:
        return result

    for f in SCALAR_FIELDS:
        result.scalar_scores[f] = scalar_field_score(parsed.get(f), ground_truth.get(f))

    for f in LIST_FIELDS:
        result.list_scores[f] = list_field_score(parsed.get(f), ground_truth.get(f))

    result.nested_scores["experience"] = nested_list_field_score(
        parsed.get("experience"), ground_truth.get("experience"),
        key_fields=["title", "company", "start_date", "end_date"])
    result.nested_scores["education"] = nested_list_field_score(
        parsed.get("education"), ground_truth.get("education"),
        key_fields=["degree", "institution", "year"])

    result.hallucinations = detect_hallucinations(parsed, source_text)

    result.exact_match = (
        schema["valid"]
        and all(v == 1 for v in result.scalar_scores.values())
        and all(s["f1"] == 1.0 for s in result.list_scores.values())
        and all(s["f1"] == 1.0 for s in result.nested_scores.values())
        and len(result.hallucinations) == 0
    )
    return result


def run_evaluation(model, tokenizer, test_examples: List[Dict[str, Any]], max_new_tokens: int = 300) -> Dict[str, Any]:
    """test_examples: list of {"instruction", "input", "output"} dicts (output = ground truth dict)."""
    import torch

    results = []
    model.eval()
    for i, ex in enumerate(test_examples):
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gt = ex["output"] if isinstance(ex["output"], dict) else json.loads(ex["output"])
        results.append(evaluate_example(i, ex["input"], raw, gt))

    return aggregate_results(results)


def aggregate_results(results: List[ExampleResult]) -> Dict[str, Any]:
    n = len(results)
    valid_json_rate = sum(r.valid_json for r in results) / n
    exact_match_rate = sum(r.exact_match for r in results) / n

    scalar_avg = {}
    for f in SCALAR_FIELDS:
        vals = [r.scalar_scores.get(f) for r in results if f in r.scalar_scores]
        scalar_avg[f] = sum(vals) / len(vals) if vals else None

    list_avg = {}
    for f in LIST_FIELDS:
        f1s = [r.list_scores[f]["f1"] for r in results if f in r.list_scores]
        list_avg[f] = sum(f1s) / len(f1s) if f1s else None

    nested_avg = {}
    for f in NESTED_LIST_FIELDS:
        f1s = [r.nested_scores[f]["f1"] for r in results if f in r.nested_scores]
        nested_avg[f] = sum(f1s) / len(f1s) if f1s else None

    total_halluc = sum(len(r.hallucinations) for r in results)
    hallucination_rate = total_halluc / n

    return {
        "n_examples": n,
        "valid_json_rate": round(valid_json_rate, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "scalar_field_accuracy": {k: (round(v, 4) if v is not None else None) for k, v in scalar_avg.items()},
        "list_field_f1": {k: (round(v, 4) if v is not None else None) for k, v in list_avg.items()},
        "nested_field_f1": {k: (round(v, 4) if v is not None else None) for k, v in nested_avg.items()},
        "hallucination_rate_per_example": round(hallucination_rate, 4),
        "per_example": [r.__dict__ for r in results],
    }


def print_report(report: Dict[str, Any]):
    print("=" * 60)
    print(f"Evaluated on {report['n_examples']} examples")
    print(f"Valid JSON/structure rate : {report['valid_json_rate']*100:.1f}%")
    print(f"Full exact-match rate     : {report['exact_match_rate']*100:.1f}%")
    print(f"Hallucinations / example  : {report['hallucination_rate_per_example']:.2f}")
    print("-" * 60)
    print("Scalar field accuracy:")
    for k, v in report["scalar_field_accuracy"].items():
        print(f"  {k:12s}: {v*100:.1f}%" if v is not None else f"  {k:12s}: n/a")
    print("List field F1:")
    for k, v in report["list_field_f1"].items():
        print(f"  {k:12s}: {v*100:.1f}%" if v is not None else f"  {k:12s}: n/a")
    print("Nested list field F1:")
    for k, v in report["nested_field_f1"].items():
        print(f"  {k:12s}: {v*100:.1f}%" if v is not None else f"  {k:12s}: n/a")
    print("=" * 60)


def check_regression(new_report: Dict[str, Any], baseline_report: Dict[str, Any], tolerance: float = 0.02) -> bool:
    """Returns True if new_report passes (no metric regressed beyond tolerance vs baseline)."""
    keys_to_check = ["valid_json_rate", "exact_match_rate"]
    passed = True
    for k in keys_to_check:
        if new_report[k] < baseline_report[k] - tolerance:
            print(f"REGRESSION on {k}: {new_report[k]} < baseline {baseline_report[k]} - {tolerance}")
            passed = False
    if new_report["hallucination_rate_per_example"] > baseline_report["hallucination_rate_per_example"] + tolerance:
        print("REGRESSION on hallucination_rate_per_example")
        passed = False
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True, help="JSONL file with instruction/input/output rows")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter, optional")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--baseline_report", default=None, help="Path to a prior report JSON for regression check")
    parser.add_argument("--output_report", default="eval_report.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")

    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)

    with open(args.test_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if args.max_examples:
        examples = examples[: args.max_examples]

    report = run_evaluation(model, tokenizer, examples)
    print_report(report)

    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.output_report}")

    if args.baseline_report:
        with open(args.baseline_report) as f:
            baseline = json.load(f)
        ok = check_regression(report, baseline)
        if not ok:
            print("\nREGRESSION CHECK FAILED")
            sys.exit(1)
        print("\nRegression check passed.")
