from nanofable.flops import flops


def test_flops_formula():
    assert flops(10, 100) == 6000


def test_flops_scales_linearly():
    assert flops(2_000_000, 500_000_000) == 6.0 * 2_000_000 * 500_000_000
