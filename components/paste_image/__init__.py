import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "paste_image",
    path=os.path.dirname(os.path.abspath(__file__)),
)


def paste_image_component(key="paste_image"):
    """粘贴图片组件。返回 JSON 字符串 '{"base64": "...", "mime": "..."}' 或 None。"""
    return _component_func(key=key, default=None)
