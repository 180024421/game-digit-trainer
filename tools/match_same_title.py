"""双图固定区域标题比对（极速版）。

优化：
- 只加载识别模型（不加载检测/方向分类）→ 冷启动约 0.4s
- 两张裁切一次 batch 推理 → 约 30~50ms
- 可选常驻服务：首次加载后，后续请求约几十毫秒

用法：
  python tools/match_same_title.py a.png b.png
  python tools/match_same_title.py --serve          # 常驻
  python tools/match_same_title.py a.png b.png      # 自动连常驻，否则本地跑
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FIXED_ROI_A = (100, 10, 200, 40)
FIXED_ROI_B = (280, 5, 240, 45)
# 小于该尺寸视为「已裁好的标题条」，不再套全屏 ROI
SMALL_CROP_MAX_H = 96
SMALL_CROP_MAX_W = 480
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765

_REC = None


def _norm(s: str) -> str:
    """只保留汉字，去掉星号/符号/空格等 OCR 噪声。"""
    s = (s or "").replace("　", "")
    chars = re.findall(r"[\u4e00-\u9fff]+", s)
    return "".join(chars)


def _is_small_crop(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    return h <= SMALL_CROP_MAX_H or w <= SMALL_CROP_MAX_W


def _resolve_roi(
    img: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    *,
    force_full: bool = False,
) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    if force_full or roi is None or _is_small_crop(img):
        return (0, 0, w, h)
    return roi


def _get_rec():
    """仅加载 PP-OCRv4 识别模型。"""
    global _REC
    if _REC is not None:
        return _REC
    from pathlib import Path as P

    import rapidocr_onnxruntime
    from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
    from rapidocr_onnxruntime.utils import read_yaml, update_model_path

    root = P(rapidocr_onnxruntime.__file__).resolve().parent
    cfg = update_model_path(read_yaml(root / "config.yaml"))
    # 限制线程，避免 CPU 抢占导致抖动
    cfg["Rec"]["intra_op_num_threads"] = 2
    cfg["Rec"]["inter_op_num_threads"] = 2
    _REC = TextRecognizer(cfg["Rec"])
    return _REC


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读图失败: {path}")
    return img


def _crop(img: np.ndarray, xywh: tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, ww, hh = xywh
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    ww = max(1, min(int(ww), w - x))
    hh = max(1, min(int(hh), h - y))
    return img[y : y + hh, x : x + ww]


def _prep_rec_img(bgr: np.ndarray) -> np.ndarray:
    """小图加边距并放大，避免贴边裁切导致漏字。"""
    pad = 8
    bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h, w = bgr.shape[:2]
    # 标题条高度通常很小，放大到约 64px 高再识别更稳
    if h < 64:
        scale = 64.0 / h
        bgr = cv2.resize(
            bgr,
            (max(8, int(round(w * scale))), 64),
            interpolation=cv2.INTER_CUBIC,
        )
    elif max(h, w) < 160:
        bgr = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return bgr


def _parse_xywh(s: str, default: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    raw = (s or "").strip().lower()
    if not raw:
        return default
    if raw in {"full", "0", "none", "auto"}:
        return None
    parts = [int(float(p.strip())) for p in raw.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit(f"ROI 需 x,y,w,h 或 full，收到: {s}")
    return parts[0], parts[1], parts[2], parts[3]


def _decide(text_a: str, text_b: str, keyword: str) -> tuple[bool, str]:
    kw = _norm(keyword)
    if kw:
        matched = (kw in text_a) and (kw in text_b)
        reason = (
            f"两边都含「{kw}」"
            if matched
            else f"关键词未双边命中（关键词「{kw}」，实际 A={text_a!r} B={text_b!r}）"
        )
        return matched, reason
    if not text_a or not text_b:
        return False, f"两边文本不一致：A={text_a!r} B={text_b!r}"
    if text_a == text_b or text_a in text_b or text_b in text_a:
        return True, f"两边文本一致：{text_a!r} / {text_b!r}"
    # 较长公共子串（防星号/漏字导致不完全相等）
    shorter, longer = (text_a, text_b) if len(text_a) <= len(text_b) else (text_b, text_a)
    if len(shorter) >= 3 and shorter in longer:
        return True, f"两边核心一致：{shorter!r}"
    best = 0
    for i in range(len(shorter)):
        for j in range(i + 3, len(shorter) + 1):
            sub = shorter[i:j]
            if sub in longer:
                best = max(best, len(sub))
    if best >= max(3, min(len(text_a), len(text_b)) - 1):
        return True, f"两边近似匹配（公共 {best} 字）：A={text_a!r} B={text_b!r}"
    return False, f"两边文本不一致：A={text_a!r} B={text_b!r}"


def match_titles(
    path_a: Path,
    path_b: Path,
    *,
    keyword: str = "",
    roi_a: tuple[int, int, int, int] | None = FIXED_ROI_A,
    roi_b: tuple[int, int, int, int] | None = FIXED_ROI_B,
    force_full: bool = False,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    img_a = _read_bgr(path_a)
    img_b = _read_bgr(path_b)
    used_a = _resolve_roi(img_a, roi_a, force_full=force_full)
    used_b = _resolve_roi(img_b, roi_b, force_full=force_full)
    crop_a = _prep_rec_img(_crop(img_a, used_a))
    crop_b = _prep_rec_img(_crop(img_b, used_b))

    t_load = time.perf_counter()
    rec = _get_rec()
    load_ms = (time.perf_counter() - t_load) * 1000

    t_ocr = time.perf_counter()
    results, _elapse = rec([crop_a, crop_b])
    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    text_a = _norm(results[0][0]) if results and results[0] else ""
    conf_a = float(results[0][1]) if results and results[0] else 0.0
    text_b = _norm(results[1][0]) if results and len(results) > 1 and results[1] else ""
    conf_b = float(results[1][1]) if results and len(results) > 1 and results[1] else 0.0

    matched, reason = _decide(text_a, text_b, keyword)
    return {
        "matched": matched,
        "reason": reason,
        "text_a": text_a,
        "text_b": text_b,
        "conf_a": round(conf_a, 4),
        "conf_b": round(conf_b, 4),
        "roi_a": list(used_a),
        "roi_b": list(used_b),
        "full_a": used_a == (0, 0, img_a.shape[1], img_a.shape[0]),
        "full_b": used_b == (0, 0, img_b.shape[1], img_b.shape[0]),
        "load_ms": round(load_ms, 1),
        "ocr_ms": round(ocr_ms, 1),
        "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
        "via": "local",
    }


def _server_alive(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.05):
            return True
    except OSError:
        return False


def match_via_server(
    path_a: Path,
    path_b: Path,
    *,
    keyword: str = "",
    roi_a: tuple[int, int, int, int] | None = FIXED_ROI_A,
    roi_b: tuple[int, int, int, int] | None = FIXED_ROI_B,
    force_full: bool = False,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> dict[str, Any]:
    import urllib.request

    payload = json.dumps(
        {
            "a": str(Path(path_a).resolve()),
            "b": str(Path(path_b).resolve()),
            "keyword": keyword,
            "roi_a": list(roi_a) if roi_a else None,
            "roi_b": list(roi_b) if roi_b else None,
            "force_full": force_full,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/match",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data["client_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    data["via"] = "server"
    return data


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("[serve] " + (fmt % args) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/match":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw.decode("utf-8"))

            def _roi(key: str, default):
                v = req.get(key, default)
                if v is None:
                    return None
                return tuple(v)

            rep = match_titles(
                Path(req["a"]),
                Path(req["b"]),
                keyword=str(req.get("keyword") or ""),
                roi_a=_roi("roi_a", FIXED_ROI_A),
                roi_b=_roi("roi_b", FIXED_ROI_B),
                force_full=bool(req.get("force_full")),
            )
            body = json.dumps(rep, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    print(f"加载识别模型中…", flush=True)
    t0 = time.perf_counter()
    _get_rec()
    print(f"模型就绪 {((time.perf_counter()-t0)*1000):.0f}ms · http://{host}:{port}/match", flush=True)
    # 热身
    dummy = np.zeros((40, 160, 3), dtype=np.uint8)
    _get_rec()([dummy, dummy])
    httpd = HTTPServer((host, port), _Handler)
    print("常驻服务已启动，Ctrl+C 结束。客户端会自动连这里。", flush=True)
    httpd.serve_forever()


def _print_report(rep: dict[str, Any], *, as_json: bool = False) -> None:
    flag = "MATCH" if rep.get("matched") else "NO_MATCH"
    print(
        f"{flag} | A=「{rep.get('text_a')}」({float(rep.get('conf_a') or 0):.0%}) "
        f"B=「{rep.get('text_b')}」({float(rep.get('conf_b') or 0):.0%}) "
        f"| ocr={rep.get('ocr_ms')}ms total={rep.get('total_ms') or rep.get('client_ms')}ms "
        f"via={rep.get('via')}"
    )
    print(rep.get("reason") or "")
    if as_json:
        print(json.dumps(rep, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="固定区域标题极速比对")
    ap.add_argument("image_a", type=Path, nargs="?")
    ap.add_argument("image_b", type=Path, nargs="?")
    ap.add_argument("--keyword", default="", help="可选关键词；默认只比 A/B 文本是否一致")
    ap.add_argument("--roi-a", default="", help="像素 x,y,w,h；小图可写 full；默认自动")
    ap.add_argument("--roi-b", default="", help="像素 x,y,w,h；小图可写 full；默认自动")
    ap.add_argument("--full", action="store_true", help="强制整图识别（适合已裁好的标题条）")
    ap.add_argument("--serve", action="store_true", help="启动常驻服务（推荐）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--local", action="store_true", help="强制本地跑，不连常驻服务")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.serve:
        run_server(port=args.port)
        return 0

    if not args.image_a or not args.image_b:
        ap.print_help()
        return 2

    roi_a = _parse_xywh(args.roi_a, None if args.full else FIXED_ROI_A)
    roi_b = _parse_xywh(args.roi_b, None if args.full else FIXED_ROI_B)
    force_full = bool(args.full)

    if args.debug:
        debug = Path("_match_debug")
        debug.mkdir(exist_ok=True)
        for path, roi, name in (
            (args.image_a, roi_a, "crop_a.png"),
            (args.image_b, roi_b, "crop_b.png"),
        ):
            img = _read_bgr(path)
            used = _resolve_roi(img, roi, force_full=force_full)
            crop = _crop(img, used)
            cv2.imencode(".png", crop)[1].tofile(str(debug / name))

    use_server = (not args.local) and _server_alive(port=args.port)
    if use_server:
        try:
            rep = match_via_server(
                args.image_a,
                args.image_b,
                keyword=args.keyword,
                roi_a=roi_a,
                roi_b=roi_b,
                force_full=force_full,
                port=args.port,
            )
        except Exception as exc:
            print(f"常驻服务失败，回退本地: {exc}", file=sys.stderr)
            rep = match_titles(
                args.image_a,
                args.image_b,
                keyword=args.keyword,
                roi_a=roi_a,
                roi_b=roi_b,
                force_full=force_full,
            )
    else:
        rep = match_titles(
            args.image_a,
            args.image_b,
            keyword=args.keyword,
            roi_a=roi_a,
            roi_b=roi_b,
            force_full=force_full,
        )
        if not args.local:
            print(
                "提示: 先开「比对标题服务.cmd」常驻，之后每次比对可到几十毫秒。",
                file=sys.stderr,
            )

    _print_report(rep, as_json=args.json)
    return 0 if rep.get("matched") else 1


if __name__ == "__main__":
    raise SystemExit(main())
