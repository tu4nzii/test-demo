"""Model calls for bubble charts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp

from gemini_calls import FAILURE_TEXT, chat_with_gemini

from ...common.chart_io import image_to_data_url
from ...common.json_utils import parse_model_json, unwrap_openai_content
from ...common.model_config import get_chat_completion_urls, get_model_name

from .parser import extract_coords


JSON_ONLY_SYSTEM_PROMPT = (
    "You are a precise chart-reading assistant. Always return only valid JSON. "
    "Do not include Markdown fences, explanations, or prose."
)

DEFAULT_LEGACY_URLS = [
    "http://localhost:8200/v1/chat/completions",
    "http://localhost:8201/v1/chat/completions",
    "http://localhost:8202/v1/chat/completions",
    "http://localhost:8203/v1/chat/completions",
    "http://localhost:8204/v1/chat/completions",
    "http://localhost:8205/v1/chat/completions",
    "http://localhost:8100/v1/chat/completions",
    "http://localhost:8101/v1/chat/completions",
    "http://localhost:8102/v1/chat/completions",
    "http://localhost:8103/v1/chat/completions",
    "http://localhost:8104/v1/chat/completions",
    "http://localhost:8300/v1/chat/completions",
    "http://localhost:8301/v1/chat/completions",
    "http://localhost:8302/v1/chat/completions",
    "http://localhost:8303/v1/chat/completions",
    "http://localhost:8304/v1/chat/completions",
]


class PointModelClient:
    def __init__(
        self,
        legacy_urls: list[str] | None = None,
        *,
        max_retries: int = 3,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> None:
        self.urls = get_chat_completion_urls(legacy_urls or DEFAULT_LEGACY_URLS)
        self.model_name = get_model_name()
        self.max_retries = max_retries
        self.timeout = timeout or aiohttp.ClientTimeout(total=300, connect=30, sock_connect=30, sock_read=240)
        self._session: aiohttp.ClientSession | None = None
        self._index = 0

    async def __aenter__(self) -> "PointModelClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    def _next_url(self) -> str:
        url = self.urls[self._index]
        self._index = (self._index + 1) % len(self.urls)
        return url

    def _payload(self, prompt: str, image_path: Path) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": 512,
            "max_completion_tokens": 512,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    async def call_text(self, prompt: str, image_path: Path, label: str) -> str | None:
        url = self._next_url()
        print(f"[bubble model] {label} -> {url}")
        request_urls = [url] + [item for item in self.urls if item != url]
        content = await chat_with_gemini(
            self._payload(prompt, image_path)["messages"],
            model=self.model_name,
            max_tokens=512,
            response_format={"type": "json_object"},
            extra_payload={"max_completion_tokens": 512},
            temperature=0,
            urls=request_urls,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return None if content == FAILURE_TEXT else content

    async def predict_coords(self, prompt: str, image_path: Path, point_name: str) -> tuple[Any, Any]:
        content = await self.call_text(prompt, image_path, point_name)
        if not content:
            return (-1, -1)
        parsed = parse_model_json(content)
        return extract_coords(parsed, point_name)

    async def check_exists(self, prompt: str, image_path: Path) -> bool:
        content = await self.call_text(prompt, image_path, "exists")
        if not content:
            return False
        parsed = parse_model_json(content)
        if isinstance(parsed, dict) and "exists" in parsed:
            return bool(parsed["exists"])
        lowered = content.strip().lower()
        return "yes" in lowered and "no" not in lowered
