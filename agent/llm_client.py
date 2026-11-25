"""OpenAI-compatible chat completions client with native tool calling, retries, and token accounting."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config


@dataclass
class LLMResponse:
    content: str
    total_tokens: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_content: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


class LLMClient:
    """Thin stdlib client for OpenAI-compatible Chat Completions APIs."""

    def __init__(
        self,
        api_key: str = config.LLM_API_KEY,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kw: Any,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError(
                "API key missing: set the DEEPSEEK_API_KEY environment variable or local .env file."
            )
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kw.get("temperature", config.LLM_TEMPERATURE),
            "max_tokens": kw.get("max_tokens", config.LLM_MAX_TOKENS),
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = kw.get("tool_choice", "auto")
            body["parallel_tool_calls"] = kw.get("parallel_tool_calls", False)

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last: Exception = RuntimeError("unreachable")
        for attempt in range(config.LLM_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                raw_tcs = msg.get("tool_calls") or []
                norm_tcs: List[Dict[str, Any]] = []
                for idx, tc in enumerate(raw_tcs):
                    fn = tc.get("function") or {}
                    norm_tcs.append({
                        "id": str(tc.get("id") or f"call_{idx}"),
                        "type": "function",
                        "function": {
                            "name": str(fn.get("name", "")),
                            "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str)
                            else json.dumps(fn.get("arguments") or {}, ensure_ascii=False),
                        },
                    })
                raw_usage = data.get("usage") or {}
                total_tok = int(raw_usage.get("total_tokens", 0))
                usage_dict = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                    "total_tokens": total_tok,
                }
                return LLMResponse(
                    content=msg.get("content") or "",
                    total_tokens=total_tok,
                    tool_calls=norm_tcs,
                    reasoning_content=msg.get("reasoning_content") or "",
                    usage=usage_dict,
                )
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                err = RuntimeError(f"HTTP {e.code} from {self.base_url} (model={self.model}): {detail or e.reason}")
                last = err
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise err from e
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {config.LLM_RETRIES} retries: {last}")
