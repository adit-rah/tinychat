"""Pluggable coherence-judge backends.

The frozen default judge is `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0), loaded 4-bit on a 16GB
GPU. `AnthropicJudge` is a paid fallback for borderline configs (spec §8). `parse_judge_json`
robustly extracts the three integer scores from a model's free-form text and clamps to [0, 5].
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

AXES = ("grammar", "consistency", "completes")
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "judge_prompt.md")


def load_judge_prompt() -> str:
    """Extract the fenced prompt template from judge_prompt.md."""
    text = open(_PROMPT_PATH).read()
    m = re.search(r"```(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def fill_prompt(template: str, prefix: str, completion: str) -> str:
    """Fill {prefix}/{completion} by plain replacement — NOT str.format, which chokes on
    the literal JSON example braces in the frozen prompt (KeyError: '"grammar"')."""
    return template.replace("{prefix}", prefix).replace("{completion}", completion)


def _clamp(v) -> int:
    return max(0, min(5, int(round(float(v)))))


def parse_judge_json(text: str) -> dict:
    """Extract {grammar, consistency, completes} from messy judge output; clamp to 0..5.

    Tolerates surrounding prose, code fences, and trailing tokens by scanning for the first
    JSON object. Missing axes default to 0.
    """
    obj = None
    for match in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            cand = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if any(a in cand for a in AXES):
            obj = cand
            break
    if obj is None:
        obj = {}
    return {a: _clamp(obj.get(a, 0)) for a in AXES}


def per_completion_score(scores: dict) -> float:
    """Mean of the three axes — the per-completion coherence score (spec §8 rubric)."""
    return sum(scores[a] for a in AXES) / len(AXES)


class JudgeBackend(Protocol):
    def score(self, prefix: str, completion: str) -> dict: ...


class LocalQwenJudge:
    """Frozen default: Qwen2.5-7B-Instruct via transformers (4-bit on GPU by default)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", four_bit: bool = True,
                 max_new_tokens: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.template = load_judge_prompt()
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Pin fp16 everywhere: without an explicit dtype the non-quantized modules (and
        # load-time staging) can materialize at fp32 and OOM a 16GB card.
        kwargs = {"device_map": "auto", "dtype": torch.float16}
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def score(self, prefix: str, completion: str) -> dict:
        prompt = fill_prompt(self.template, prefix, completion)
        messages = [{"role": "user", "content": prompt}]
        # return_dict=True: on transformers>=5 the template call yields a BatchEncoding
        # either way, and generate() only accepts it unpacked, not positionally.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
        return parse_judge_json(text)


class AnthropicJudge:
    """Paid fallback judge (Anthropic API) for borderline configs (spec §8)."""

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.template = load_judge_prompt()

    def score(self, prefix: str, completion: str) -> dict:
        prompt = fill_prompt(self.template, prefix, completion)
        msg = self.client.messages.create(
            model=self.model, max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_judge_json(msg.content[0].text)
