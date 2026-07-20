"""金标回归集：固定截图 + 期望字符串，导出后一键自检。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from game_digit_trainer.gold import compare_preds, tokenize_expected
from game_digit_trainer.labels import display_label
from game_digit_trainer.predict import predict_boxes_string, predict_image_string
from game_digit_trainer.preprocess import load_bgr
from game_digit_trainer.project import GameProject


def regression_dir(project: GameProject) -> Path:
    d = project.root / "regression"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cases_path(project: GameProject) -> Path:
    return regression_dir(project) / "cases.json"


def load_cases(project: GameProject) -> list[dict[str, Any]]:
    path = cases_path(project)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("cases") or [])
    except Exception:
        return []


def save_cases(project: GameProject, cases: list[dict[str, Any]]) -> None:
    path = cases_path(project)
    path.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_regression_case(
    project: GameProject,
    *,
    image_path: Path,
    expected: str,
    boxes: list[tuple[int, int, int, int]] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """把当前截图+金标加入回归集。"""
    reg = regression_dir(project)
    cases = load_cases(project)
    stem = name or f"case_{len(cases) + 1:03d}"
    dest = reg / f"{stem}{Path(image_path).suffix or '.png'}"
    shutil.copy2(image_path, dest)
    item: dict[str, Any] = {
        "name": stem,
        "image": dest.name,
        "expected": expected.strip(),
        "boxes": [list(b) for b in (boxes or [])],
    }
    if roi is not None:
        item["roi"] = [int(v) for v in roi]
    cases.append(item)
    save_cases(project, cases)
    return item


def run_regression(project: GameProject, checkpoint: Path) -> dict[str, Any]:
    """跑全部回归用例（单字 checkpoint），返回汇总。"""
    cases = load_cases(project)
    results: list[dict[str, Any]] = []
    ok_n = 0
    for case in cases:
        img = regression_dir(project) / str(case.get("image") or "")
        expected_raw = str(case.get("expected") or "")
        if not img.is_file():
            results.append({"name": case.get("name"), "ok": False, "error": "缺图"})
            continue
        try:
            boxes_raw = case.get("boxes") or []
            boxes = [tuple(int(x) for x in b) for b in boxes_raw]  # type: ignore[misc]
            if boxes:
                bgr = load_bgr(img)
                text, parts = predict_boxes_string(project, bgr, boxes, checkpoint)
            else:
                text, parts = predict_image_string(project, img, checkpoint)
            expected = tokenize_expected(expected_raw, project.config.classes)
            pred_labels = [p[0] for p in parts]
            diffs = compare_preds(expected, [(a, b) for a, b in parts])
            exp_disp = "".join(display_label(x) for x in expected)
            passed = text == exp_disp or (not diffs and len(pred_labels) == len(expected))
            if passed:
                ok_n += 1
            results.append(
                {
                    "name": case.get("name"),
                    "ok": passed,
                    "expected": exp_disp,
                    "got": text,
                    "diffs": len(diffs),
                }
            )
        except Exception as exc:
            results.append({"name": case.get("name"), "ok": False, "error": str(exc)})
    total = len(cases)
    return {
        "total": total,
        "passed": ok_n,
        "failed": total - ok_n,
        "results": results,
        "mode": "char",
    }


def run_line_regression(project: GameProject, checkpoint: Path) -> dict[str, Any]:
    """用行模型跑回归：优先 case.roi，否则整图。"""
    from game_digit_trainer.predict_line import predict_line_path

    cases = load_cases(project)
    results: list[dict[str, Any]] = []
    ok_n = 0
    for case in cases:
        img = regression_dir(project) / str(case.get("image") or "")
        expected_raw = str(case.get("expected") or "")
        if not img.is_file():
            results.append({"name": case.get("name"), "ok": False, "error": "缺图"})
            continue
        try:
            roi = None
            raw_roi = case.get("roi")
            if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
                roi = tuple(int(x) for x in raw_roi)  # type: ignore[assignment]
            elif case.get("boxes") and len(case["boxes"]) == 1:
                b = case["boxes"][0]
                if len(b) == 4:
                    roi = tuple(int(x) for x in b)  # type: ignore[assignment]
            text, _parts, _conf = predict_line_path(project, img, checkpoint, region=roi)
            expected = tokenize_expected(expected_raw, project.config.classes)
            exp_disp = "".join(display_label(x) for x in expected)
            passed = text == exp_disp
            if passed:
                ok_n += 1
            results.append(
                {
                    "name": case.get("name"),
                    "ok": passed,
                    "expected": exp_disp,
                    "got": text,
                    "diffs": 0 if passed else 1,
                }
            )
        except Exception as exc:
            results.append({"name": case.get("name"), "ok": False, "error": str(exc)})
    total = len(cases)
    return {
        "total": total,
        "passed": ok_n,
        "failed": total - ok_n,
        "results": results,
        "mode": "line",
    }
