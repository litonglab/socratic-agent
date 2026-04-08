import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "chat_input_images",
    path=os.path.dirname(os.path.abspath(__file__)),
)


def chat_input_images(key="chat_input_images"):
    """ChatGPT-style chat input with image paste/upload and thumbnails.

    Returns JSON string '{"text": "...", "images": [{"base64": "...", "mime": "..."}, ...]}' or None.
    """
    return _component_func(key=key, default=None)
