from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="game-digit-trainer",
        description="游戏 HUD 多字体数字：切字 / 修正 / 训练 / 导出 ONNX",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gui", help="打开图形界面")

    c = sub.add_parser("create", help="新建游戏项目")
    c.add_argument("game_id")
    c.add_argument("--symbols", action="store_true", help="启用 , / % :")
    c.add_argument("--units", action="store_true", help="启用 万/亿")
    c.add_argument("--base", type=Path, default=None, help="projects 父目录，默认 cwd")

    s = sub.add_parser("segment", help="对图片切字写入 pending")
    s.add_argument("--project", type=Path, required=True)
    s.add_argument("--image", type=Path, required=True)
    s.add_argument("--invert", action="store_true")

    t = sub.add_parser("train", help="训练单字 CNN")
    t.add_argument("--project", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=15)

    tl = sub.add_parser("train-line", help="训练行模型 CRNN（推荐）")
    tl.add_argument("--project", type=Path, required=True)
    tl.add_argument("--epochs", type=int, default=12)
    tl.add_argument("--no-finetune", action="store_true", help="从头训练，不续训")
    tl.add_argument("--bootstrap", action="store_true", help="开启行→单字自动粗切（默认关）")
    tl.add_argument("--no-augment", action="store_true")

    e = sub.add_parser("export", help="导出单字 ONNX 包")
    e.add_argument("--project", type=Path, required=True)
    e.add_argument("--run", type=Path, default=None, help="best.pt 路径，默认最新")

    el = sub.add_parser("export-line", help="导出行 ONNX（digits_line.onnx）")
    el.add_argument("--project", type=Path, required=True)
    el.add_argument("--run", type=Path, default=None, help="line_best.pt，默认最新")

    pr = sub.add_parser("predict", help="单字模型试推理一张图")
    pr.add_argument("--project", type=Path, required=True)
    pr.add_argument("--image", type=Path, required=True)
    pr.add_argument("--run", type=Path, default=None)

    pl = sub.add_parser("predict-line", help="行模型试推理（整图或需自行裁好）")
    pl.add_argument("--project", type=Path, required=True)
    pl.add_argument("--image", type=Path, required=True)
    pl.add_argument("--run", type=Path, default=None)

    b = sub.add_parser("bootstrap-chars", help="从行样本粗切单字入库")
    b.add_argument("--project", type=Path, required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "gui":
        from game_digit_trainer.gui import run_gui

        return run_gui()

    if args.cmd == "create":
        from game_digit_trainer.project import create_project

        proj = create_project(
            args.game_id, args.base, with_symbols=args.symbols, with_units=args.units
        )
        print(proj.root)
        return 0

    if args.cmd == "segment":
        from game_digit_trainer.project import open_project
        from game_digit_trainer.segment import save_pending_chars, segment_image

        proj = open_project(args.project)
        if args.invert:
            proj.config.preprocess.invert = True
        _, crops, _ = segment_image(args.image, proj.config.preprocess)
        paths = save_pending_chars(proj, args.image, crops)
        print(f"pending={len(paths)}")
        return 0

    if args.cmd == "train":
        from game_digit_trainer.project import open_project
        from game_digit_trainer.train import train_project

        proj = open_project(args.project)
        path = train_project(proj, epochs=args.epochs)
        print(path)
        return 0

    if args.cmd == "train-line":
        from game_digit_trainer.project import open_project
        from game_digit_trainer.train_line import train_line_project

        proj = open_project(args.project)
        path = train_line_project(
            proj,
            epochs=args.epochs,
            finetune=not args.no_finetune,
            auto_bootstrap=bool(args.bootstrap),
            augment_real=not args.no_augment,
            log=print,
        )
        print(path)
        return 0

    if args.cmd == "export":
        from game_digit_trainer.export_onnx import export_onnx, latest_checkpoint
        from game_digit_trainer.project import open_project

        proj = open_project(args.project)
        ckpt = args.run or latest_checkpoint(proj)
        if not ckpt:
            print("无 checkpoint", file=sys.stderr)
            return 2
        out = export_onnx(proj, Path(ckpt))
        print(out.parent)
        return 0

    if args.cmd == "export-line":
        from game_digit_trainer.export_line_onnx import export_line_onnx
        from game_digit_trainer.project import open_project
        from game_digit_trainer.train_line import latest_line_checkpoint

        proj = open_project(args.project)
        ckpt = args.run or latest_line_checkpoint(proj)
        if not ckpt:
            print("无行模型 checkpoint", file=sys.stderr)
            return 2
        out = export_line_onnx(proj, Path(ckpt))
        print(out)
        return 0

    if args.cmd == "predict":
        from game_digit_trainer.export_onnx import latest_checkpoint
        from game_digit_trainer.predict import predict_image_string
        from game_digit_trainer.project import open_project

        proj = open_project(args.project)
        ckpt = args.run or latest_checkpoint(proj)
        if not ckpt:
            print("无 checkpoint", file=sys.stderr)
            return 2
        text, parts = predict_image_string(proj, args.image, Path(ckpt))
        print(text)
        for lab, conf in parts:
            print(f"  {lab} {conf:.3f}")
        return 0

    if args.cmd == "predict-line":
        from game_digit_trainer.predict_line import predict_line_path
        from game_digit_trainer.project import open_project
        from game_digit_trainer.train_line import latest_line_checkpoint

        proj = open_project(args.project)
        ckpt = args.run or latest_line_checkpoint(proj)
        if not ckpt:
            print("无行模型", file=sys.stderr)
            return 2
        text, parts, conf = predict_line_path(proj, args.image, Path(ckpt))
        print(text)
        print(f"conf={conf:.3f}")
        for lab, c in parts:
            print(f"  {lab} {c:.3f}")
        return 0

    if args.cmd == "bootstrap-chars":
        from game_digit_trainer.line_data import bootstrap_chars_from_line_samples
        from game_digit_trainer.project import open_project

        proj = open_project(args.project)
        added, previews = bootstrap_chars_from_line_samples(proj)
        print(f"added={sum(added.values())} detail={added} preview={len(previews)}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
