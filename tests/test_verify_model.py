from pathlib import Path

import numpy as np

from game_digit_trainer.predict import list_project_models, resolve_onnx_pack
from game_digit_trainer.project import create_project


def test_resolve_onnx_pack_file_and_dir(tmp_path: Path):
    pack = tmp_path / "export_a"
    pack.mkdir()
    onnx = pack / "digits.onnx"
    onnx.write_bytes(b"fake")
    (pack / "digits.labels").write_text("0\n1\n", encoding="utf-8")
    (pack / "manifest.json").write_text(
        '{"input":{"width":32,"height":32}}', encoding="utf-8"
    )

    ref = resolve_onnx_pack(onnx)
    assert ref.kind == "onnx"
    assert ref.path == onnx.resolve()
    assert ref.labels_path and ref.labels_path.is_file()

    ref2 = resolve_onnx_pack(pack)
    assert ref2.path == onnx.resolve()


def test_list_project_models_includes_exports_and_recent(tmp_path: Path):
    proj = create_project("verify_scan", base=tmp_path)
    exp = proj.exports_dir
    exp.mkdir(parents=True, exist_ok=True)
    onnx = exp / "digits.onnx"
    onnx.write_bytes(b"x")
    (exp / "digits.labels").write_text("0\n", encoding="utf-8")

    external = tmp_path / "other_pc"
    external.mkdir()
    e_onnx = external / "digits.onnx"
    e_onnx.write_bytes(b"y")
    (external / "digits.labels").write_text("0\n1\n", encoding="utf-8")

    refs = list_project_models(proj, recent_onnx=[str(e_onnx)])
    kinds = {r.kind for r in refs}
    assert "onnx" in kinds
    paths = {str(r.path.resolve()) for r in refs}
    assert str(onnx.resolve()) in paths
    assert str(e_onnx.resolve()) in paths
