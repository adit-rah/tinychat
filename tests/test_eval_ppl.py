import math

import torch

from nanofable.data import batch_iterator
from nanofable.model import build_model
from nanofable.train import evaluate_ppl


def test_ppl_is_exp_mean_ce(data_paths, small_cfg):
    _, val_path = data_paths
    model = build_model(small_cfg, "fp16")
    ctx = small_cfg.ctx

    ppl = evaluate_ppl(model, val_path, ctx, n_batches=2, eval_rows=16, seed=1234)

    # Recompute the same two batches by hand and confirm ppl == exp(mean CE).
    it = batch_iterator(val_path, ctx, 16 * ctx, seed=1234)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(2):
            x, y = next(it)
            _, loss = model(x, y)
            losses.append(loss.item())
    expected = math.exp(sum(losses) / len(losses))

    assert ppl > 1.0 and math.isfinite(ppl)
    assert abs(ppl - expected) < 1e-4
