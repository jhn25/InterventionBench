"""OpenAI-compatible HTTP client (works for OpenAI, Anthropic via compat shim, Kimi, etc.)."""

from __future__ import annotations

import time

import httpx

from interventionbench.models.base import ChatMessage, ModelClient, ModelResponse


class OpenAICompatibleClient(ModelClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict | None = None,
    ) -> ModelResponse:
        body: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if extra_body:
            body.update(extra_body)

        last_err = None
        # Retry on BOTH network errors (ConnectError, ReadTimeout, RemoteProtocolError)
        # AND HTTP 5xx / 429. The old code only retried HTTP 5xx, so any transient
        # `Connection refused` (vLLM queue full, restart, brief network blip) killed
        # the cell immediately — and surfaced as [MODEL ERROR] [Errno 111] in the
        # worker. With multiple workers hammering one vLLM endpoint, this happens
        # often enough to corrupt thousands of cells per run.
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
                last_err = f"network: {type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return ModelResponse(content=content, usage=usage, raw=data)
            if (500 <= resp.status_code < 600 or resp.status_code == 429) and attempt < self.max_retries:
                time.sleep(min(2**attempt, 30))
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue
            raise RuntimeError(f"API call failed: HTTP {resp.status_code}: {resp.text[:500]}")

        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")
