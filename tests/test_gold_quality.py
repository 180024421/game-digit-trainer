from game_digit_trainer.gold import compare_preds, tokenize_expected
from game_digit_trainer.quality import min_samples_gate


def test_tokenize_expected():
    classes = [str(i) for i in range(10)] + ["dot", "wan", "colon"]
    assert tokenize_expected("12万", classes) == ["1", "2", "wan"]
    assert tokenize_expected("1.2万", classes) == ["1", "dot", "2", "wan"]
    assert tokenize_expected("2:03", classes) == ["2", "colon", "0", "3"]


def test_compare_preds():
    mm = compare_preds(["1", "2"], [("1", 0.9), ("3", 0.5)])
    assert len(mm) == 1
    assert mm[0]["index"] == 1


def test_min_samples_gate():
    errs = min_samples_gate({str(i): 10 for i in range(10)})
    assert errs == []
    errs = min_samples_gate({"0": 1, "1": 10})
    assert any("不足" in e for e in errs)
