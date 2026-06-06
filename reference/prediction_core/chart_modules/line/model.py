"""Model calls for line charts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp

from reference.prediction_core.chart_io import image_to_data_url
from reference.prediction_core.json_utils import parse_model_json, unwrap_openai_content
from reference.prediction_core.model_config import get_chat_completion_urls, get_headers, get_model_name

from .parser import extract_coords


class LineModelClient:
    def __init__(
        self,
        legacy_urls: list[str] | None = None,
        *,
        max_retries: int = 3,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> None:
        self.urls = get_chat_completion_urls(legacy_urls)
        self.model_name = get_model_name()
        self.max_retries = max_retries
        self.timeout = timeout or aiohttp.ClientTimeout(total=300, connect=30, sock_connect=30, sock_read=240)
        self._session: aiohttp.ClientSession | None = None
        self._index = 0

    async def __aenter__(self) -> "LineModelClient":
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
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": 512,
        }

    async def call_text(self, prompt: str, image_path: Path, label: str) -> str | None:
        session = await self._ensure_session()
        for attempt in range(1, self.max_retries + 1):
            url = self._next_url()
            print(f"[line model] {label} -> {url}")
            try:
                async with session.post(url, headers=get_headers(), json=self._payload(prompt, image_path)) as resp:
                    text = await asyncio.wait_for(resp.text(), timeout=90)
                    if resp.status != 200:
                        print(f"[line model] HTTP {resp.status}: {text[:200]}")
                        await asyncio.sleep(2 * attempt)
                        continue
                    return unwrap_openai_content(text)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                print(f"[line model] attempt {attempt}/{self.max_retries} failed: {exc}")
                await asyncio.sleep(2 * attempt)
        return None

    async def predict_coords(self, prompt: str, image_path: Path, point_name: str) -> tuple[Any, Any]:
        content = await self.call_text(prompt, image_path, point_name)
        if not content:
            return ("", -1)
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
