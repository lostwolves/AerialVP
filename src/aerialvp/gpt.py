"""Chat Completions client for configurable text and vision endpoints."""
from __future__ import annotations

import ast
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

import requests


class GPTClient:
    def __init__(self, config: dict, model: str):
        self.config = config
        self.model = model

    def chat(self, question: str, system_prompt: str | None = None,
             image_files: list[str] | str | None = None) -> str:
        key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
        if not key or key.strip().upper() == "XXX":
            raise RuntimeError("Set OPENAI_API_KEY in the project .env file or process environment.")
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt or "You are a useful assistant who can efficiently complete user-specified tasks or answer user questions well."}]}]
        content = [{"type": "text", "text": question}]
        paths = [image_files] if isinstance(image_files, str) else (image_files or [])
        for name in paths:
            path = Path(name)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        messages.append({"role": "user", "content": content})
        payload = {"model": self.model, "temperature": self.config.get("temperature", 0.1),
                   "top_p": self.config.get("top_p", 0.9), "max_tokens": self.config.get("max_tokens", 4096),
                   "messages": messages}
        attempts = max(1, int(self.config.get("max_retries", 2)))
        for attempt in range(attempts):
            try:
                response = requests.post(self.config["url"], json=payload,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                                         timeout=self.config.get("timeout", 180))
            except (requests.Timeout, requests.ConnectionError):
                if attempt + 1 == attempts:
                    raise RuntimeError(f"Chat Completions connection failed for {self.model} after {attempts} attempts") from None
                time.sleep(2)
                continue
            if response.status_code != 200:
                if response.status_code in (408, 429, 500, 502, 503, 504) and attempt + 1 < attempts:
                    time.sleep(2)
                    continue
                # Do not include provider response bodies, which may echo request contents.
                raise RuntimeError(f"Chat Completions HTTP {response.status_code} for model {self.model}")
            try:
                answer = response.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                raise RuntimeError("Provider returned an invalid Chat Completions response") from None
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Provider returned empty or non-text message content")
            return answer
        raise RuntimeError("Chat Completions failed")


def parse_object(text: str) -> dict:
    """Accept JSON or the dictionary notation printed in the paper, never execute it."""
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed response code fence")
        value = "\n".join(lines[1:-1])
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError("Expected one JSON object with no extra prose") from None
    if not isinstance(obj, dict):
        raise ValueError("Expected a dictionary/object")
    return obj

