import torch
from transformers import BatchEncoding

from eval.judge import (
    AXES,
    LocalQwenJudge,
    fill_prompt,
    load_judge_prompt,
    parse_judge_json,
    per_completion_score,
)


class FakeJudge:
    """Implements the JudgeBackend protocol without any model download."""

    def score(self, prefix: str, completion: str) -> dict:
        return {"grammar": 5, "consistency": 4, "completes": 4}


def test_parse_clean_json():
    s = '{"grammar": 5, "consistency": 4, "completes": 3}'
    assert parse_judge_json(s) == {"grammar": 5, "consistency": 4, "completes": 3,
                                   "parsed": True}


def test_parse_messy_text_with_prose_and_fence():
    s = 'Sure! Here is my rating:\n```json\n{"grammar": 4, "consistency": 5, "completes": 5}\n```\nThanks.'
    assert parse_judge_json(s) == {"grammar": 4, "consistency": 5, "completes": 5,
                                   "parsed": True}


def test_parse_clamps_out_of_range():
    s = '{"grammar": 9, "consistency": -2, "completes": 3}'
    assert parse_judge_json(s) == {"grammar": 5, "consistency": 0, "completes": 3,
                                   "parsed": True}


def test_parse_missing_axis_defaults_zero():
    s = '{"grammar": 5}'
    assert parse_judge_json(s) == {"grammar": 5, "consistency": 0, "completes": 0,
                                   "parsed": True}


def test_parse_failure_is_flagged_not_silent():
    # No JSON at all, and JSON truncated before the closing brace (the max_new_tokens
    # failure mode): both must come back parsed=False so zeros are auditable.
    for s in ("I cannot rate this story.",
              'Here is my rating:\n{"grammar": 4, "consistency": 5, "compl'):
        out = parse_judge_json(s)
        assert out["parsed"] is False
        assert all(out[a] == 0 for a in AXES)


def test_fill_frozen_prompt_survives_json_example():
    # The frozen prompt ends with a literal JSON example in braces; filling the template
    # must not choke on it (str.format raises KeyError: '"grammar"').
    out = fill_prompt(load_judge_prompt(), "Once upon a time", "they lived happily.")
    assert "Once upon a time" in out
    assert "they lived happily." in out
    assert '{"grammar":' in out  # JSON example preserved verbatim
    assert "{prefix}" not in out and "{completion}" not in out


class FakeV5Tokenizer:
    """transformers>=5: apply_chat_template returns a BatchEncoding, not a raw tensor."""

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            return_tensors=None, return_dict=False):
        assert return_dict, "judge must pass return_dict=True to be version-stable"
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]]),
                              "attention_mask": torch.tensor([[1, 1, 1]])})

    def decode(self, ids, skip_special_tokens=True):
        assert ids.tolist() == [9, 9]  # only the newly generated tokens
        return '{"grammar": 4, "consistency": 3, "completes": 5}'


class FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        assert torch.is_tensor(input_ids)  # BatchEncoding passed positionally would crash
        return torch.cat([input_ids, torch.tensor([[9, 9]])], dim=1)


def test_local_judge_score_handles_v5_batchencoding():
    # Regression: generate(BatchEncoding) raised AttributeError on transformers 5.x.
    j = LocalQwenJudge.__new__(LocalQwenJudge)  # skip __init__ (no model download)
    j.template = load_judge_prompt()
    j.max_new_tokens = 64
    j.tokenizer = FakeV5Tokenizer()
    j.model = FakeModel()
    out = j.score("Once", "upon a time")
    assert {a: out[a] for a in AXES} == {"grammar": 4, "consistency": 3, "completes": 5}
    assert out["parsed"] is True
    assert out["raw"] == '{"grammar": 4, "consistency": 3, "completes": 5}'


def test_fake_judge_protocol_shape():
    j = FakeJudge()
    out = j.score("prefix", "completion")
    assert set(out) == set(AXES)
    assert per_completion_score(out) == (5 + 4 + 4) / 3
