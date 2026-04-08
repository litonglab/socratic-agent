"""图片文字识别模块：通过 OCR 提取图片中的文本信息。"""
from __future__ import annotations

import io
import threading

_ocr_engine = None
_ocr_lock = threading.Lock()


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        with _ocr_lock:
            if _ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR
                _ocr_engine = RapidOCR()
    return _ocr_engine


def describe_image(
    image_bytes: bytes,
    filename: str = "image.png",
    user_text: str = "",
) -> str:
    """
    使用 OCR 提取图片中的文字。

    Args:
        image_bytes: 原始图片字节。
        filename: 文件名（未使用，保留接口兼容）。
        user_text: 未使用，保留接口兼容。

    Returns:
        图片中识别出的文字内容。
    """
    try:
        ocr = _get_ocr()
        result, _ = ocr(image_bytes)
        if not result:
            return "[图片中未识别到文字]"
        lines = [item[1] for item in result]
        return "\n".join(lines)
    except Exception as e:
        return f"[OCR 识别失败: {e}]"
