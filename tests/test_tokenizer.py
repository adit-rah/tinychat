from nanofable.tokenizer import BOS, EOS, PAD, load_tokenizer, train_tokenizer

CORPUS = [
    "Once upon a time there was a little cat named Tom.",
    "Tom liked to play in the garden every day.",
    "One day Tom found a red ball and was very happy.",
    "The sun was warm and the sky was blue.",
    "Lily and Tom played together until the evening.",
] * 50


def test_roundtrip_ascii(tmp_path):
    path = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, save_path=path, vocab_size=512)
    tok = load_tokenizer(path)
    s = "Once upon a time there was a little cat named Tom."
    assert tok.decode(tok.encode(s).ids) == s


def test_vocab_size_capped(tmp_path):
    path = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, save_path=path, vocab_size=512)
    tok = load_tokenizer(path)
    assert tok.get_vocab_size() <= 512


def test_special_token_ids_stable(tmp_path):
    path = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, save_path=path, vocab_size=512)
    tok = load_tokenizer(path)
    assert tok.token_to_id(BOS) == 0
    assert tok.token_to_id(EOS) == 1
    assert tok.token_to_id(PAD) == 2
