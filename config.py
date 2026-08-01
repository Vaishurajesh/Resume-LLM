"""
config.py — Centralized configuration for the Resume Intelligence inference service.
Reads from environment variables with sensible defaults, so behavior can be
changed without editing code (12-factor app style).
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    base_model_name: str = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path: str = os.environ.get("ADAPTER_PATH", "./resume-lora-adapter")
    use_adapter: bool = os.environ.get("USE_ADAPTER", "true").lower() == "true"
    device: str = os.environ.get("DEVICE", "auto")  # "auto", "cuda", "cpu"
    dtype: str = os.environ.get("MODEL_DTYPE", "float16")  # "float16", "bfloat16", "float32"
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "300"))
    max_input_chars: int = int(os.environ.get("MAX_INPUT_CHARS", "8000"))
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")


SYSTEM_INSTRUCTION = (
    "Extract structured resume information from the input text. "
    "Return ONLY valid JSON matching the schema: name, email, phone, "
    "location, summary, skills (list), experience (list of {title, company, "
    "start_date, end_date, description}), education (list of {degree, "
    "institution, year}), certifications (list). Use null for missing "
    "fields and [] for missing lists. Do not invent information that is "
    "not present in the input."
)

REQUIRED_SCHEMA_KEYS = {
    "name", "email", "phone", "location", "summary",
    "skills", "experience", "education", "certifications",
}

config = Config()
