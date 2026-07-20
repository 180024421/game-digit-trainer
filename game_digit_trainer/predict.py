from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from game_digit_trainer.model import DigitCNN
from game_digit_trainer.preprocess import apply_preprocess, load_bgr
from game_digit_trainer.project import GameProject
from game_digit_trainer.segment import prepare_tensor_image, segment_binary


def load_checkpoint(path: Path) -> tuple[DigitCNN, list[str], int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(ckpt["classes"])
    w = int(ckpt["input_width"])
    h = int(ckpt["input_height"])
    model = DigitCNN(num_classes=len(classes), in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes, w, h


def predict_char(
    model: DigitCNN,
    classes: list[str],
    gray: np.ndarray,
    width: int,
    height: int,
) -> tuple[str, float]:
    arr = prepare_tensor_image(gray, width, height)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax().item())
        conf = float(prob[idx].item())
    return classes[idx], conf


def predict_image_string(
    project: GameProject,
    image_path: Path,
    checkpoint: Path,
    *,
    conf_threshold: float = 0.5,
) -> tuple[str, list[tuple[str, float]]]:
    model, classes, w, h = load_checkpoint(checkpoint)
    bgr = load_bgr(image_path)
    binary = apply_preprocess(bgr, project.config.preprocess)
    crops = segment_binary(binary)
    parts: list[tuple[str, float]] = []
    from game_digit_trainer.labels import display_label

    for crop in crops:
        label, conf = predict_char(model, classes, crop.image, w, h)
        parts.append((label, conf))
    if any(c < conf_threshold for _, c in parts):
        text = "".join("?" if c < conf_threshold else display_label(l) for l, c in parts)
    else:
        text = "".join(display_label(l) for l, _ in parts)
    return text, parts


def predict_pending_file(
    checkpoint: Path,
    pending: Path,
    classes: list[str],
    width: int,
    height: int,
) -> tuple[str, float]:
    model, ck_classes, w, h = load_checkpoint(checkpoint)
    use_classes = ck_classes or classes
    raw = np.fromfile(str(pending), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取: {pending}")
    return predict_char(model, use_classes, img, w or width, h or height)


def predict_boxes_string(
    project: GameProject,
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    checkpoint: Path,
    *,
    conf_threshold: float = 0.5,
) -> tuple[str, list[tuple[str, float]]]:
    """按手动字框顺序推理整行字符串。"""
    from game_digit_trainer.labels import display_label
    from game_digit_trainer.segment import crops_from_full_boxes

    model, classes, w, h = load_checkpoint(checkpoint)
    crops = crops_from_full_boxes(bgr, boxes, project.config.preprocess)
    parts: list[tuple[str, float]] = []
    for crop in crops:
        label, conf = predict_char(model, classes, crop.image, w, h)
        parts.append((label, conf))
    text = "".join(
        "?" if c < conf_threshold else display_label(l) for l, c in parts
    )
    return text, parts


def score_pending_files(
    checkpoint: Path,
    paths: list[Path],
    classes: list[str],
    width: int,
    height: int,
) -> list[tuple[Path, str, float]]:
    """批量预标待审文件，返回 (path, label, conf)。"""
    if not paths:
        return []
    model, ck_classes, w, h = load_checkpoint(checkpoint)
    use_classes = ck_classes or classes
    out: list[tuple[Path, str, float]] = []
    for pending in paths:
        try:
            raw = np.fromfile(str(pending), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            lab, conf = predict_char(model, use_classes, img, w or width, h or height)
            out.append((pending, lab, conf))
        except Exception:
            continue
    return out


def check_onnx_dependency() -> tuple[bool, str]:
    try:
        import onnx  # noqa: F401

        return True, f"onnx {getattr(onnx, '__version__', '')}"
    except ImportError:
        return False, "未安装 onnx。请运行: pip install onnx onnxscript"


def check_onnxruntime_dependency() -> tuple[bool, str]:
    try:
        import onnxruntime as ort  # noqa: F401

        return True, f"onnxruntime {getattr(ort, '__version__', '')}"
    except ImportError:
        return False, "未安装 onnxruntime。请运行: pip install onnxruntime"


@dataclass(frozen=True)
class ModelRef:
    """可选推理模型：导出 ONNX 包或 PyTorch checkpoint。"""

    kind: str  # "onnx" | "pt"
    path: Path
    labels_path: Path | None = None
    manifest_path: Path | None = None
    display: str = ""

    def key(self) -> str:
        return f"{self.kind}:{self.path.resolve()}"


def resolve_onnx_pack(path: Path) -> ModelRef:
    """接受 digits.onnx 文件或含该文件的目录。"""
    p = path.expanduser().resolve()
    if p.is_dir():
        onnx = p / "digits.onnx"
        if not onnx.is_file():
            # 兼容子目录仅有一个 .onnx
            onnxs = list(p.glob("*.onnx"))
            if len(onnxs) == 1:
                onnx = onnxs[0]
            else:
                raise ValueError(f"目录中找不到 digits.onnx: {p}")
    elif p.suffix.lower() == ".onnx":
        onnx = p
    else:
        raise ValueError("请选择 digits.onnx 或导出目录")
    if not onnx.is_file():
        raise ValueError(f"找不到模型文件: {onnx}")
    labels = onnx.parent / "digits.labels"
    if not labels.is_file():
        raise ValueError(f"同目录缺少 digits.labels: {onnx.parent}")
    manifest = onnx.parent / "manifest.json"
    parent_name = onnx.parent.name
    display = f"[ONNX] {parent_name}/{onnx.name}"
    return ModelRef(
        kind="onnx",
        path=onnx,
        labels_path=labels,
        manifest_path=manifest if manifest.is_file() else None,
        display=display,
    )


def list_project_models(
    project: GameProject | None,
    *,
    recent_onnx: list[str] | None = None,
    max_pt: int = 10,
) -> list[ModelRef]:
    """扫描工程 exports/runs 与最近使用的外部 ONNX。"""
    out: list[ModelRef] = []
    seen: set[str] = set()

    def _add(ref: ModelRef) -> None:
        k = ref.key()
        if k in seen:
            return
        seen.add(k)
        out.append(ref)

    if project is not None:
        exp = project.exports_dir
        if exp.is_dir():
            for onnx in sorted(exp.rglob("digits.onnx"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    _add(resolve_onnx_pack(onnx))
                except ValueError:
                    continue
        if project.runs_dir.is_dir():
            bests = sorted(
                project.runs_dir.glob("*/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:max_pt]
            for pt in bests:
                _add(
                    ModelRef(
                        kind="pt",
                        path=pt,
                        display=f"[PT] {pt.parent.name}/{pt.name}",
                    )
                )

    for raw in recent_onnx or []:
        try:
            _add(resolve_onnx_pack(Path(raw)))
        except (ValueError, OSError):
            continue
    return out


def _read_onnx_meta(ref: ModelRef) -> tuple[list[str], int, int]:
    assert ref.labels_path is not None
    classes = [
        ln.strip()
        for ln in ref.labels_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not classes:
        raise ValueError("digits.labels 为空")
    w, h = 32, 32
    if ref.manifest_path and ref.manifest_path.is_file():
        import json

        man = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
        inp = man.get("input") or {}
        w = int(inp.get("width") or w)
        h = int(inp.get("height") or h)
    return classes, w, h


def predict_char_onnx(
    session: object,
    classes: list[str],
    gray: np.ndarray,
    width: int,
    height: int,
) -> tuple[str, float]:
    arr = prepare_tensor_image(gray, width, height)
    x = arr[np.newaxis, np.newaxis, ...].astype(np.float32)
    input_name = session.get_inputs()[0].name  # type: ignore[attr-defined]
    logits = session.run(None, {input_name: x})[0][0]  # type: ignore[attr-defined]
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    exp = np.exp(logits)
    prob = exp / exp.sum()
    idx = int(prob.argmax())
    return classes[idx], float(prob[idx])


def predict_boxes_string_onnx(
    project: GameProject,
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    onnx_ref: ModelRef | Path,
    *,
    conf_threshold: float = 0.5,
) -> tuple[str, list[tuple[str, float]]]:
    ok, msg = check_onnxruntime_dependency()
    if not ok:
        raise RuntimeError(msg)
    import onnxruntime as ort

    ref = onnx_ref if isinstance(onnx_ref, ModelRef) else resolve_onnx_pack(Path(onnx_ref))
    if ref.kind != "onnx":
        raise ValueError("需要 ONNX 模型")
    classes, w, h = _read_onnx_meta(ref)
    sess = ort.InferenceSession(str(ref.path), providers=["CPUExecutionProvider"])
    from game_digit_trainer.labels import display_label
    from game_digit_trainer.segment import crops_from_full_boxes

    crops = crops_from_full_boxes(bgr, boxes, project.config.preprocess)
    parts: list[tuple[str, float]] = []
    for crop in crops:
        label, conf = predict_char_onnx(sess, classes, crop.image, w, h)
        parts.append((label, conf))
    text = "".join("?" if c < conf_threshold else display_label(l) for l, c in parts)
    return text, parts


def predict_boxes_with_model(
    project: GameProject,
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    model: ModelRef,
    *,
    conf_threshold: float = 0.5,
) -> tuple[str, list[tuple[str, float]]]:
    if model.kind == "onnx":
        return predict_boxes_string_onnx(
            project, bgr, boxes, model, conf_threshold=conf_threshold
        )
    return predict_boxes_string(
        project, bgr, boxes, model.path, conf_threshold=conf_threshold
    )

