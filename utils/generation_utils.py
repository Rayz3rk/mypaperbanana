# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Unified task-scoped model calls, with compatibility aliases for older tools.

No process-global model clients or credentials are mutated by a browser session.
"""

import os
from pathlib import Path

import yaml

from utils.model_session import (GenerationError, ImageBudgetExceeded, call_text_async,
                                 call_image_async, has_model_session, model_session)


def get_config_val(section, key, env_var, default=""):
    value = os.getenv(env_var)
    if value:
        return value
    path = Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return (values or {}).get(section, {}).get(key) or default


def _convert_to_gemini_parts(contents):
    """Kept for external scripts; both historical image structures are supported."""
    from google.genai import types
    from providers.openai_compatible import decode_data_url, image_url_from_content
    parts = []
    for item in contents:
        if item.get("type") == "text":
            parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") in {"image", "image_url"}:
            url = image_url_from_content(item)
            if url and url.startswith("data:"):
                data, mime = decode_data_url(url)
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
            elif url:
                raise ValueError("此兼容转换器仅接受内联图片；请使用 GeminiProvider 处理远端图片。")
    return parts


async def call_evolink_text_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=2, error_context=""):
    return await call_text_async(model_name, contents, config, max_attempts, retry_delay, error_context)


async def call_evolink_image_with_retry_async(model_name, prompt, config, max_attempts=1, retry_delay=2, error_context=""):
    return await call_image_async(model_name, prompt, config, max_attempts, retry_delay, error_context)


async def call_gemini_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=2, error_context=""):
    values = config if isinstance(config, dict) else config.model_dump(exclude_none=True)
    modalities = values.get("response_modalities") or []
    if any(str(value).upper().endswith("IMAGE") for value in modalities):
        from providers.openai_compatible import image_url_from_content
        prompt = "\n".join(item.get("text", "") for item in contents if item.get("type") == "text")
        references = [image_url_from_content(item) for item in contents if item.get("type") in {"image", "image_url"}]
        image_config = values.get("image_config") or {}
        return await call_image_async(model_name, prompt, {
            "aspect_ratio": image_config.get("aspect_ratio", "16:9"),
            "quality": image_config.get("image_size", "2K"), "image_urls": [value for value in references if value]},
            max_attempts, retry_delay, error_context)
    return await call_text_async(model_name, contents, values, max_attempts, retry_delay, error_context)


async def call_openai_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=2, error_context=""):
    values = dict(config)
    values.setdefault("max_output_tokens", values.get("max_completion_tokens", 8192))
    count = int(values.get("candidate_num", 1))
    results = []
    for _ in range(count):
        results.extend(await call_text_async(model_name, contents, values, max_attempts, retry_delay, error_context))
    return results


async def call_claude_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=2, error_context=""):
    # Claude and other models can be selected through OpenRouter/New API or another
    # compatible connection; evaluation uses the same isolated text connection.
    return await call_openai_with_retry_async(model_name, contents, config, max_attempts, retry_delay, error_context)


async def call_openai_image_generation_with_retry_async(model_name, prompt, config, max_attempts=1, retry_delay=2, error_context=""):
    return await call_image_async(model_name, prompt, config, max_attempts, retry_delay, error_context)


async def upload_image_to_evolink(image_b64: str, media_type: str = "image/png") -> str:
    """Adapters upload if necessary; never upload through another session's key."""
    return f"data:{media_type};base64,{image_b64}"
