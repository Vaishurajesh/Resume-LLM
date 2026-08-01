# Part 1 — Research & Model Selection
## Replacing GPT-4 Mini for a Resume Intelligence Platform

### 1. Task Framing

Resume Intelligence means: given raw resume text (PDF-extracted, often messy),
produce structured JSON — name, contact info, education, work experience,
skills, certifications — while handling missing fields, inconsistent layouts,
multi-column PDFs, and non-English content gracefully.

This is fundamentally a **structured-extraction / information-extraction**
task, not open-ended generation. That reframes what "good" means: we need a
model that is obedient to schema, cheap to run at high volume (resumes are
processed in bulk), and easy to fine-tune on our own labeled data — not a
model with the broadest general reasoning ability.

### 2. Chosen Model: **Qwen2.5-1.5B-Instruct**

| Attribute | Value |
|---|---|
| Parameters | 1.54B |
| Context window | 32,768 tokens (native), extendable via YaRN |
| License | Apache 2.0 (fully permissive, commercial use allowed) |
| Architecture | Dense transformer, GQA, RoPE, RMSNorm |
| Tokenizer | ~150k vocab BPE, good multilingual coverage |
| Hardware (inference) | Runs in fp16 on a single T4/L4 (~4GB VRAM); runs in 4-bit on CPU/edge |
| Hardware (LoRA fine-tune) | Single consumer GPU (RTX 3090/4090, 24GB) or free-tier Colab T4 (16GB) with QLoRA |

**Why this model:**

1. **Right-sized for the task.** Resume field extraction doesn't need
   70B-scale reasoning; it needs reliable pattern-following and JSON
   discipline. A 1.5B instruction-tuned model, fine-tuned on a few thousand
   resume examples, reliably outperforms a zero-shot much larger model on
   this narrow task, at a fraction of the latency and cost.
2. **Apache 2.0 license.** No usage restrictions, no attribution
   requirements that complicate a commercial SaaS product — unlike Llama's
   custom license (monthly-active-user clause) or Gemma's usage terms.
3. **Strong structured-output behavior out of the box.** The Qwen2.5
   instruct series was explicitly trained with function-calling / JSON-mode
   style data, so it starts from a good prior for schema-constrained
   generation, meaning less fine-tuning data is needed to reach production
   quality.
4. **Cheap to serve at scale.** Resume platforms process thousands of
   documents per hour. A 1.5B model can be served on commodity GPUs (or
   even CPU with quantization) at a fraction of the cost of a 7B+ model,
   which matters directly to unit economics when replacing an
   API-metered model like GPT-4 Mini.
5. **Easy to fine-tune iteratively.** Small size means fast LoRA training
   loops (minutes, not hours), so the team can retrain frequently as new
   resume formats/edge cases are discovered — critical for a product that
   will keep encountering novel layouts.

### 3. Pros & Cons

**Pros**
- Apache 2.0, no legal friction for commercial deployment.
- Small enough for cheap, low-latency inference and fast fine-tuning cycles.
- Good multilingual tokenizer (useful since resumes come in many languages).
- 32k context comfortably covers even long multi-page resumes.
- Strong ecosystem support (vLLM, llama.cpp, Ollama, HF `transformers`,
  `peft`/QLoRA all support it day one).

**Cons**
- Lower ceiling on general reasoning than 7B+ models — e.g. resolving highly
  ambiguous cases (inferring seniority from vague titles, complex date-range
  disambiguation) may need more explicit fine-tuning data than a larger
  model would.
- Without fine-tuning, zero-shot JSON reliability is decent but not
  bulletproof — occasional schema drift is expected on truly out-of-domain
  formats.
- Smaller model = more sensitive to fine-tuning data quality; noisy labels
  hurt it more than they would a larger, more robust base model.

### 4. Why Not Other Popular Models

| Model | Why not chosen |
|---|---|
| **Llama-3.2-3B-Instruct** | Custom Meta license restricts use above 700M MAU without a separate agreement, and requires attribution ("Built with Llama"). Slightly larger with no clear extraction-quality advantage over Qwen2.5 at this scale. |
| **Mistral-7B-Instruct** | Good model, but 7B is 4-5x the inference/fine-tuning cost for a task that doesn't need the extra reasoning capacity. Apache 2.0 license is fine, but the cost/latency tradeoff doesn't pay for itself here. |
| **Gemma-2-2B** | Google's usage license has more restrictive redistribution terms than Apache 2.0, and Gemma's context window (8k) is tight for long multi-page resumes with cover letters attached. |
| **Phi-3-mini (3.8B)** | Very strong reasoning-per-parameter, MIT licensed — a legitimate alternative. Passed over mainly because it's ~2.5x the parameter count of Qwen2.5-1.5B for marginal gains on a narrow extraction task, and Qwen's JSON/function-calling training data is a closer match to our output format. |
| **GPT-4 Mini itself (closed)** | Explicitly out of scope — the assignment is to replace a hosted API with something we own, fine-tune, and control the cost/latency/privacy profile of. |

### 5. Fallback / Escalation Path

If evaluation later shows Qwen2.5-1.5B under-performing on complex edge cases
(e.g., heavily creative resume layouts, infographic-style resumes converted
to garbled text), the natural escalation path is **Qwen2.5-3B-Instruct**
(same family, same license, same tooling — only the checkpoint changes),
avoiding an architecture/tooling migration.
