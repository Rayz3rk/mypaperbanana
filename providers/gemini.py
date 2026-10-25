"""Google Gemini native REST adapter with a task-local HTTP session."""

import asyncio
import base64
from typing import Any
from urllib.parse import quote

from .connections import ModelConnection
from .openai_compatible import OpenAICompatibleProvider, ProviderRequestError, decode_data_url, image_url_from_content


class GeminiProvider(OpenAICompatibleProvider):
    def __init__(self, connection: ModelConnection):
        super().__init__(connection)
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3] + "/v1beta"
        elif not self.base_url.endswith("/v1beta"):
            self.base_url += "/v1beta"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.connection.api_key}

    async def list_models(self) -> list[str]:
        self.connection.validate(require_model=False)
        models: set[str] = set()
        page_token = ""
        for _ in range(20):
            path = "models?pageSize=100" + (f"&pageToken={quote(page_token, safe='')}" if page_token else "")
            response = await self._request_json("GET", path)
            for item in response.get("models", []):
                if isinstance(item, dict) and item.get("name") and "generateContent" in item.get("supportedGenerationMethods", ["generateContent"]):
                    models.add(str(item["name"]).removeprefix("models/"))
            page_token = response.get("nextPageToken", "")
            if not page_token:
                break
        return sorted(models)

    async def _parts(self, contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts = []
        for item in contents:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") in {"image", "image_url", "input_image"}:
                value = image_url_from_content(item)
                if not value:
                    raise ProviderRequestError("参考图片缺少数据。", stage="input")
                if value.startswith("data:"):
                    data, mime = decode_data_url(value)
                else:
                    data, mime = await self._download_bytes(value), "image/png"
                parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")}})
        return parts

    def _record_tokens(self, body: dict[str, Any]) -> None:
        metadata = body.get("usageMetadata") or {}
        for source, target in {"promptTokenCount": "prompt_tokens", "candidatesTokenCount": "completion_tokens", "totalTokenCount": "total_tokens"}.items():
            value = metadata.get(source)
            if isinstance(value, (int, float)) and value >= 0:
                self.usage[target] += int(value)

    async def generate_text(self, model_name: str, contents: list[dict[str, Any]], system_prompt: str = "",
                            temperature: float = 1.0, max_output_tokens: int = 50000, max_attempts: int = 3,
                            retry_delay: float = 5, error_context: str = "") -> list[str]:
        self.connection.validate("text", require_model=not bool(model_name))
        model = (model_name or self.connection.model).removeprefix("models/")
        payload: dict[str, Any] = {"contents": [{"role": "user", "parts": await self._parts(contents)}],
                                   "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        for attempt in range(max(1, max_attempts)):
            self.usage["text_requests"] += 1
            try:
                body = await self._request_json("POST", f"models/{quote(model, safe='')}:generateContent", payload=payload)
                break
            except ProviderRequestError as exc:
                if exc.status not in {429, 500, 502, 503, 504} or attempt + 1 >= max_attempts:
                    raise
                await asyncio.sleep(min(retry_delay * 2**attempt, 30))
        self._record_tokens(body)
        candidates = body.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "\n".join(part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought"))
        if not text.strip() or text.strip() == "Error":
            raise ProviderRequestError("Gemini 没有返回有效文本，可能被安全规则阻止或模型不支持此请求。")
        return [text]

    async def generate_image(self, model_name: str, prompt: str, aspect_ratio: str = "16:9", quality: str = "2K",
                             image_urls: list[str] | None = None, max_attempts: int = 1, retry_delay: float = 1,
                             poll_interval: float = 3, error_context: str = "") -> list[str]:
        self.connection.validate("image", require_model=not bool(model_name))
        model = (model_name or self.connection.model).removeprefix("models/")
        contents = [{"type": "text", "text": prompt}] + [{"type": "image_url", "url": url} for url in image_urls or []]
        payload = {"contents": [{"role": "user", "parts": await self._parts(contents)}],
                   "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                        "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": quality}}}
        self.usage["image_requests"] += 1
        body = await self._request_json("POST", f"models/{quote(model, safe='')}:generateContent", payload=payload)
        self._record_tokens(body)
        results = []
        for candidate in body.get("candidates") or []:
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    value = inline["data"]
                    try:
                        if not base64.b64decode(value, validate=True):
                            raise ValueError
                    except ValueError as exc:
                        raise ProviderRequestError("Gemini 返回的图片数据无效。") from exc
                    results.append(value)
        if not results:
            raise ProviderRequestError("Gemini 未返回图片，请检查此模型是否支持图像生成。")
        return results
