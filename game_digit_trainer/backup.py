"""项目备份 / 标注包导入导出。"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from game_digit_trainer.project import GameProject

INCLUDE_DIRS = ("dataset", "config.json", "exports", "regression", "hard", "pending", "lines")

# 标注包：不含 runs/exports/模型，便于换机继续标、训
LABEL_PACK_KIND = "game-digit-trainer-labels"
LABEL_PACK_DIRS = ("dataset", "lines", "pending", "hard")


@dataclass
class ImportLabelsResult:
    dataset_files: int = 0
    line_files: int = 0
    line_labels: int = 0
    pending_files: int = 0
    hard_files: int = 0
    mode: str = "merge"

    def summary(self) -> str:
        return (
            f"模式={self.mode} · 单字 {self.dataset_files} · "
            f"行图 {self.line_files}（金标 {self.line_labels}）· "
            f"待审 {self.pending_files} · 难例 {self.hard_files}"
        )


def _add_tree(zf: zipfile.ZipFile, folder: Path, arc_prefix: str) -> int:
    """把目录写入 zip，返回文件数。"""
    n = 0
    if not folder.exists():
        return 0
    if folder.is_file():
        zf.write(folder, arc_prefix)
        return 1
    for path in folder.rglob("*"):
        if path.is_file():
            zf.write(path, f"{arc_prefix}/{path.relative_to(folder).as_posix()}")
            n += 1
    return n


def backup_project(project: GameProject, dest: Path | None = None) -> Path:
    """打包 dataset/config/exports/regression/hard/pending/lines 为 zip。"""
    root = project.root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dest or (root / "backups" / f"{project.config.game_id}_{stamp}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        cfg = root / "config.json"
        if cfg.is_file():
            zf.write(cfg, "config.json")
        for name in ("dataset", "exports", "regression", "hard", "pending", "lines"):
            folder = root / name
            if not folder.exists():
                continue
            if folder.is_file():
                zf.write(folder, name)
                continue
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())
    return out


def export_labels_pack(project: GameProject, dest: Path | None = None) -> Path:
    """导出标注数据包（单字库 + 行样本/待审 + 难例 + config）。"""
    root = project.root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dest or (root / "backups" / f"{project.config.game_id}_labels_{stamp}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": LABEL_PACK_KIND,
        "version": 1,
        "game_id": project.config.game_id,
        "classes": list(project.config.classes),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labels_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        cfg = root / "config.json"
        if cfg.is_file():
            zf.write(cfg, "config.json")
        for name in LABEL_PACK_DIRS:
            _add_tree(zf, root / name, name)
    return out


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    n = 1
    while True:
        cand = path.with_name(f"{stem}_imp{n}{suf}")
        if not cand.exists():
            return cand
        n += 1


def _copy_file_merge(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(dest)
    shutil.copy2(src, dest)
    return dest


def _clear_label_dirs(project: GameProject) -> None:
    """替换导入前清空标注相关目录（保留 config / runs / exports）。"""
    for name in LABEL_PACK_DIRS:
        folder = project.root / name
        if folder.is_dir():
            shutil.rmtree(folder)
        elif folder.is_file():
            folder.unlink(missing_ok=True)
    project.ensure_dirs()


def _read_jsonl_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(item.get("image") or "")
        text = str(item.get("text") or "").strip()
        if name and text:
            out[name] = text
    return out


def _write_jsonl_map(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"image": name, "text": text}, ensure_ascii=False)
        for name, text in sorted(mapping.items())
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def _pack_root(extracted: Path) -> Path:
    """兼容 zip 根目录或单层子目录包装。"""
    if (extracted / "labels_manifest.json").is_file() or (extracted / "dataset").exists() or (
        extracted / "lines"
    ).exists():
        return extracted
    subs = [p for p in extracted.iterdir() if p.is_dir()]
    if len(subs) == 1:
        return subs[0]
    return extracted


def import_labels_pack(
    project: GameProject,
    zip_path: Path,
    *,
    mode: str = "merge",
) -> ImportLabelsResult:
    """
    从标注包导入。

    mode:
      - merge：同名文件自动加 _impN，行金标按图名合并
      - replace：先清空 dataset/lines/pending/hard 再导入
    """
    if mode not in {"merge", "replace"}:
        raise ValueError("mode 应为 merge 或 replace")
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"标注包不存在: {zip_path}")

    result = ImportLabelsResult(mode=mode)
    with tempfile.TemporaryDirectory(prefix="gdt_labels_") as tmp:
        tmp_root = Path(tmp)
        _extract_zip(zip_path, tmp_root)
        src = _pack_root(tmp_root)

        man = src / "labels_manifest.json"
        if man.is_file():
            try:
                meta = json.loads(man.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"labels_manifest.json 无效: {exc}") from exc
            kind = str(meta.get("kind") or "")
            if kind and kind != LABEL_PACK_KIND:
                raise ValueError(f"不是本工具标注包（kind={kind}）")

        if mode == "replace":
            _clear_label_dirs(project)

        project.ensure_dirs()
        root = project.root

        # 单字 dataset/<class>/*
        ds = src / "dataset"
        if ds.is_dir():
            for class_dir in ds.iterdir():
                if not class_dir.is_dir():
                    continue
                dest_class = root / "dataset" / class_dir.name
                dest_class.mkdir(parents=True, exist_ok=True)
                for f in class_dir.rglob("*"):
                    if not f.is_file():
                        continue
                    if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                        continue
                    rel = f.relative_to(class_dir)
                    _copy_file_merge(f, dest_class / rel)
                    result.dataset_files += 1

        # 单字 pending
        pend = src / "pending"
        if pend.is_dir():
            dest_p = root / "pending"
            dest_p.mkdir(parents=True, exist_ok=True)
            for f in pend.iterdir():
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    _copy_file_merge(f, dest_p / f.name)
                    result.pending_files += 1

        # hard
        hard = src / "hard"
        if hard.is_dir():
            dest_h = root / "hard"
            dest_h.mkdir(parents=True, exist_ok=True)
            for f in hard.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(hard)
                    target = dest_h / rel
                    if target.name == "index.json" and target.exists() and mode == "merge":
                        # 简单覆盖 index；图片仍按 merge 拷贝
                        shutil.copy2(f, target)
                    else:
                        _copy_file_merge(f, target)
                    result.hard_files += 1

        # lines：已标图 + pending + labels.jsonl
        lines_src = src / "lines"
        if lines_src.is_dir():
            lines_dest = root / "lines"
            lines_dest.mkdir(parents=True, exist_ok=True)
            pending_src = lines_src / "pending"
            if pending_src.is_dir():
                pending_dest = lines_dest / "pending"
                pending_dest.mkdir(parents=True, exist_ok=True)
                for f in pending_src.iterdir():
                    if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        _copy_file_merge(f, pending_dest / f.name)
                        result.pending_files += 1

            name_map: dict[str, str] = {}  # old image name -> new name
            for f in lines_src.iterdir():
                if not f.is_file():
                    continue
                if f.name == "labels.jsonl":
                    continue
                if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                written = _copy_file_merge(f, lines_dest / f.name)
                name_map[f.name] = written.name
                result.line_files += 1

            incoming = _read_jsonl_map(lines_src / "labels.jsonl")
            current = {} if mode == "replace" else _read_jsonl_map(lines_dest / "labels.jsonl")
            for old_name, text in incoming.items():
                new_name = name_map.get(old_name, old_name)
                # 若图未拷到但金标在，且目标已有同名图则沿用
                if new_name not in name_map.values() and (lines_dest / old_name).is_file():
                    new_name = old_name
                if not (lines_dest / new_name).is_file():
                    continue
                current[new_name] = text
                result.line_labels += 1
            _write_jsonl_map(lines_dest / "labels.jsonl", current)

    return result
