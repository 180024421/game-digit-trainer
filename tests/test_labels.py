from game_digit_trainer.labels import label_to_index, normalize_label


def test_normalize_symbols():
    assert normalize_label(",") == "comma"
    assert normalize_label(".") == "dot"
    assert normalize_label("/") == "slash"
    assert normalize_label("3") == "3"
    assert normalize_label("万") == "wan"
    assert normalize_label("亿") == "yi"


def test_label_index():
    classes = ["0", "1", "2", "comma", "wan"]
    assert label_to_index(classes, ",") == 3
    assert label_to_index(classes, "1") == 1
    assert label_to_index(classes, "万") == 4


def test_build_class_list():
    from game_digit_trainer.labels import build_class_list

    assert "wan" not in build_class_list()
    assert "wan" in build_class_list(with_units=True)
    assert "dot" in build_class_list(with_symbols=True)
    assert "comma" in build_class_list(with_symbols=True)


def test_display_dot():
    from game_digit_trainer.labels import display_label

    assert display_label("dot") == "."
