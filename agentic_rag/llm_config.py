import inspect
import os
from typing import Optional

from langchain_deepseek import ChatDeepSeek

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"


def _pick_param(sig: inspect.Signature, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if name in sig.parameters:
            return name
    return None


def build_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0,
    streaming: bool = True,
) -> ChatDeepSeek:
    """
    Prefer DeepSeek (OpenAI-compatible) if DEEPSEEK_API_KEY is set; otherwise fall back to OpenAI.
    This keeps existing behavior runnable while enabling DeepSeek by config.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    use_deepseek = bool(deepseek_key)
    api_key = deepseek_key if use_deepseek else openai_key

    if use_deepseek:
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        model_name = model or os.getenv("DEEPSEEK_CHAT_MODEL", DEFAULT_DEEPSEEK_CHAT_MODEL)
    else:
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = model or os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL)

    sig = inspect.signature(ChatDeepSeek.__init__)
    kwargs = {"model": model_name, "temperature": temperature}

    if api_key:
        key_param = _pick_param(sig, ["api_key", "openai_api_key"])
        if key_param:
            kwargs[key_param] = api_key

    if base_url:
        base_param = _pick_param(sig, ["api_base", "base_url", "openai_api_base"])
        if base_param:
            kwargs[base_param] = base_url

    if streaming and "streaming" in sig.parameters:
        kwargs["streaming"] = True

    if os.getenv("LLM_DEBUG", "").lower() in {"1", "true", "yes"}:
        provider = "deepseek" if use_deepseek else "openai"
        base_show = base_url or "(default)"
        key_show = "set" if api_key else "missing"
        print(f"[LLM] provider={provider} model={model_name} base_url={base_show} key={key_show}")

    return ChatDeepSeek(**kwargs)
