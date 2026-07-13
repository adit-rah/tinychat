from nanofable.config import TIERS


def test_tiers_fixed_ctx_and_vocab():
    for c in TIERS.values():
        assert c.ctx == 512 and c.vocab == 4096


def test_tiny_shape_and_mlp_hidden():
    t = TIERS["tiny"]
    assert (t.n_layer, t.n_embd, t.n_head) == (4, 128, 4)
    assert t.mlp_hidden % 64 == 0


def test_head_divides_embd():
    for c in TIERS.values():
        assert c.n_embd % c.n_head == 0


def test_all_four_tiers_present():
    assert set(TIERS) == {"tiny", "small", "medium", "large"}


def test_mlp_hidden_is_eight_thirds_rounded():
    # 8/3 * 256 = 682.67 -> round up to multiple of 64 -> 704
    assert TIERS["small"].mlp_hidden == 704
