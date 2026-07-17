from game_digit_trainer.labels import label_to_index, normalize_label


def test_normalize_symbols():
    assert normalize_label(",") == "comma"
    assert normalize_label("/") == "slash"
    assert normalize_label("3") == "3"


def test_label_index():
    classes = ["0", "1", "2", "comma"]
    assert label_to_index(classes, ",") == 3
    assert label_to_index(classes, "1") == 1
