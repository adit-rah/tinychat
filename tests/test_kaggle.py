import os

import pytest

from tinychat.kaggle import Ctx, verify_frozen_artifacts

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_verify_frozen_artifacts_passes_on_repo():
    # The committed frozen set must be present and exactly 200 prefixes.
    assert verify_frozen_artifacts(REPO_ROOT) == 200


def test_verify_frozen_artifacts_fails_when_missing(tmp_path):
    with pytest.raises(AssertionError):
        verify_frozen_artifacts(str(tmp_path))


def test_ctx_is_constructible():
    c = Ctx("repo", "data", "runs", "data/train.bin", "data/val.bin", "tok.json")
    assert c.runs_dir == "runs" and c.train_path == "data/train.bin"
