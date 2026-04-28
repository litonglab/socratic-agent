import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Generator, Optional

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_CHAT_MODEL = "deepseek-chat"

load_dotenv()


def _message_to_role(msg) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "user"


def _serialize_message(msg) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "role": _message_to_role(msg),
        "content": getattr(msg, "content", ""),
    }
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        payload["tool_calls"] = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call.get("args", {}), ensure_ascii=False),
                },
            }
            for tool_call in msg.tool_calls
        ]
    if isinstance(msg, ToolMessage):
        payload["tool_call_id"] = msg.tool_call_id
        if getattr(msg, "name", None):
            payload["name"] = msg.name
    return payload


class DeepSeekChatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0,
        timeout: float = 60,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def invoke(self, messages):
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [_serialize_message(m) for m in messages],
        }
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return SimpleNamespace(content=content)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"DeepSeek API error: {last_error}")

    def invoke_stream(self, messages) -> Generator[str, None, None]:
        """流式调用 DeepSeek API，逐 token yield 内容。

        流式模式下不做重试：一旦开始 yield token，重试会导致内容重复。
        连接建立阶段的异常（如 HTTP 4xx/5xx）会直接抛出。
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": True,
            "messages": [_serialize_message(m) for m in messages],
        }
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        resp = self._session.post(
            url, json=payload, headers=headers,
            timeout=self.timeout, stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                return
            try:
                data = json.loads(data_str)
                content = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if content:
                    yield content
            except json.JSONDecodeError:
                continue


def build_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0,
    streaming: bool = False,
) -> DeepSeekChatClient:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek API.")

    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    model_name = model or os.getenv("DEEPSEEK_CHAT_MODEL", DEFAULT_DEEPSEEK_CHAT_MODEL)

    if os.getenv("LLM_DEBUG", "").lower() in {"1", "true", "yes"}:
        base_show = base_url or "(default)"
        print(f"[LLM] provider=deepseek model={model_name} base_url={base_show} key=set")

    return DeepSeekChatClient(
        api_key=deepseek_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
    )
