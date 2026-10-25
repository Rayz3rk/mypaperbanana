"""OpenAI-compatible text, Images, and multimodal image chat adapters.

Generation is submitted once. A failed image download only retries the download;
ambiguous POST failures are surfaced instead of spending the user's budget again.
"""

import asyncio
import base64
import binascii
import re
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .base import BaseProvider
from .connections import ModelConnection, normalize_base_url


class ProviderRequestError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, stage: str = "request"):
        super().__init__(message)
        self.status = status
        self.stage = stage


def decode_data_url(value: str) -> tuple[bytes, str]:
    match = re.fullmatch(r"data:(image/[a-zA-Z0-9.+-]+);base64,(.+)", value, re.DOTALL)
    if not match:
        raise ProviderRequestError("参考图片必须是图片 data URL 或 http(s) URL。", stage="input")
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ProviderRequestError("图片 base64 数据无效。", stage="input") from exc
    if not data:
        raise ProviderRequestError("图片数据为空。", stage="input")
    return data, match.group(1)


def image_url_from_content(item: dict[str, Any]) -> str | None:
    source = item.get("source") or {}
    if source.get("type") == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    if item.get("image_base64"):
        value = item["image_base64"]
        return value if value.startswith("data:") else f"data:{item.get('media_type', 'image/png')};base64,{value}"
    value = item.get("image_url")
    if isinstance(value, dict):
        return value.get("url")
    return value or item.get("url")


def contents_to_messages(contents: list[dict[str, Any]], system_prompt: str = "") -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    parts = []
    for item in contents:
        if item.get("type") == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") in {"image", "image_url", "input_image"}:
            url = image_url_from_content(item)
            if not url:
                raise ProviderRequestError("参考图片缺少图片数据。", stage="input")
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if not parts:
        raise ProviderRequestError("模型输入为空。", stage="input")
    messages.append({"role": "user", "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts})
    return messages


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, connection: ModelConnection):
        self.connection = connection
        self.base_url = normalize_base_url(connection.base_url, connection.provider, connection.image_api)
        self._session: aiohttp.ClientSession | None = None
        self.usage: dict[str, int] = {"text_requests": 0, "image_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=12))
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.connection.api_key}"}

    async def _request_json(self, method: str, path: str, *, payload: dict[str, Any] | None = None,
                            form: aiohttp.FormData | None = None) -> dict[str, Any]:
        session = await self._get_session()
        url = f"{self.base_url}/{path.lstrip('/')}"
        kwargs: dict[str, Any] = {"headers": self._headers(), "timeout": aiohttp.ClientTimeout(total=240)}
        if payload is not None:
            kwargs["json"] = payload
        if form is not None:
            kwargs["data"] = form
        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    raise ProviderRequestError(f"API 请求失败（HTTP {response.status}），请检查此连接的模型、密钥、额度和接口协议。", status=response.status)
                try:
                    body = await response.json(content_type=None)
                except (ValueError, aiohttp.ContentTypeError) as exc:
                    raise ProviderRequestError("API 返回了非 JSON 数据，请检查 Base URL 和接口协议。") from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ProviderRequestError("API 连接中断或超时。生成请求可能已被服务端接收，请先检查服务商任务记录再重试。") from exc
        if not isinstance(body, dict):
            raise ProviderRequestError("API 响应结构无效。")
        if body.get("error"):
            raise ProviderRequestError("API 返回错误，请检查此连接的模型、密钥和额度。")
        return body

    def _record_tokens(self, body: dict[str, Any]) -> None:
        usage = body.get("usage") or {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, (int, float)) and value >= 0:
                self.usage[name] += int(value)

    async def list_models(self) -> list[str]:
        self.connection.validate(require_model=False)
        body = await self._request_json("GET", "models")
        return sorted({item["id"] for item in body.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)})

    async def generate_text(self, model_name: str, contents: list[dict[str, Any]], system_prompt: str = "",
                            temperature: float = 1.0, max_output_tokens: int = 50000, max_attempts: int = 3,
                            retry_delay: float = 5, error_context: str = "") -> list[str]:
        self.connection.validate("text", require_model=not bool(model_name))
        payload = {"model": model_name or self.connection.model, "messages": contents_to_messages(contents, system_prompt),
                   "temperature": temperature, "max_tokens": max_output_tokens}
        if self.connection.provider == "deepseek" and any(item.get("type") in {"image", "image_url", "input_image"} for item in contents):
            raise ProviderRequestError("DeepSeek 文本模型不支持图片输入，请关闭参考图检索和视觉评审或选择视觉文本模型。", stage="input")
        if self.connection.provider == "openai" and payload["model"].startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload.pop("temperature")
        for attempt in range(max(1, max_attempts)):
            self.usage["text_requests"] += 1
            try:
                body = await self._request_json("POST", "chat/completions", payload=payload)
                break
            except ProviderRequestError as exc:
                if exc.status not in {429, 500, 502, 503, 504} or attempt + 1 >= max_attempts:
                    raise
                await asyncio.sleep(min(retry_delay * 2**attempt, 30))
        self._record_tokens(body)
        choices = body.get("choices") or []
        value = choices[0].get("message", {}).get("content") if choices else None
        if isinstance(value, list):
            value = "\n".join(part.get("text", "") for part in value if isinstance(part, dict) and part.get("type") in {"text", "output_text"})
        if not isinstance(value, str) or not value.strip() or value.strip() == "Error":
            raise ProviderRequestError("文本模型没有返回有效文本。")
        return [value]

    async def _download_bytes(self, url: str, max_attempts: int = 3, retry_delay: float = 1) -> bytes:
        if url.startswith("data:"):
            return decode_data_url(url)[0]
        if urlsplit(url).scheme not in {"http", "https"}:
            raise ProviderRequestError("图片下载 URL 无效。", stage="download")
        for attempt in range(max(1, max_attempts)):
            try:
                session = await self._get_session()
                # Do not forward the API key to a storage/CDN host.
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=240, sock_connect=30, sock_read=90)) as response:
                    response.raise_for_status()
                    data = await response.read()
                    if not data or len(data) > 64 * 1024 * 1024:
                        raise ValueError("Image is empty or too large")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                if attempt + 1 >= max_attempts:
                    raise ProviderRequestError("图片下载失败；已生成任务不会重新提交。请从服务商记录恢复图片。", stage="download") from exc
                await asyncio.sleep(min(retry_delay * 2**attempt, 10))
        raise ProviderRequestError("图片下载失败。", stage="download")

    @staticmethod
    def _pixel_size(aspect_ratio: str) -> str:
        try:
            width, height = (float(value) for value in aspect_ratio.split(":"))
            if width <= 0 or height <= 0:
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise ProviderRequestError("宽高比必须类似 16:9。", stage="input") from exc
        return "1536x1024" if width > height else "1024x1536" if height > width else "1024x1024"

    async def _image_form(self, model: str, prompt: str, size: str, image_urls: list[str], quality: str) -> aiohttp.FormData:
        form = aiohttp.FormData()
        for key, value in {"model": model, "prompt": prompt, "n": "1", "size": size}.items():
            form.add_field(key, value)
        if model.startswith("gpt-image"):
            form.add_field("quality", quality if quality in {"auto", "low", "medium", "high"} else "auto")
        for index, url in enumerate(image_urls):
            data = await self._download_bytes(url)
            mime = decode_data_url(url)[1] if url.startswith("data:") else "image/png"
            extension = "jpg" if mime in {"image/jpeg", "image/jpg"} else "webp" if mime == "image/webp" else "png"
            form.add_field("image[]" if len(image_urls) > 1 else "image", data, filename=f"reference_{index}.{extension}", content_type=mime)
        return form

    @staticmethod
    def _image_values(body: dict[str, Any], chat: bool) -> list[str]:
        if not chat:
            return [str(item.get("b64_json") or item.get("url")) for item in body.get("data", body.get("images", []))
                    if isinstance(item, dict) and (item.get("b64_json") or item.get("url"))]
        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        parts = list(message.get("images") or [])
        content = message.get("content")
        if isinstance(content, list):
            parts.extend(content)
        elif isinstance(content, str):
            parts.extend({"url": value} for value in re.findall(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+", content))
            parts.extend({"url": value} for value in re.findall(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", content))
        return [url for item in parts if isinstance(item, dict) for url in [image_url_from_content(item)] if url]

    async def generate_image(self, model_name: str, prompt: str, aspect_ratio: str = "16:9", quality: str = "2K",
                             image_urls: list[str] | None = None, max_attempts: int = 1, retry_delay: float = 1,
                             poll_interval: float = 3, error_context: str = "") -> list[str]:
        self.connection.validate("image", require_model=not bool(model_name))
        model = model_name or self.connection.model
        references = image_urls or []
        chat = self.connection.image_api == "chat"
        if chat:
            contents = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": url}} for url in references]
            payload = {"model": model, "messages": contents_to_messages(contents), "modalities": ["image", "text"],
                       "image_config": {"aspect_ratio": aspect_ratio, "image_size": quality}}
            path, form = "chat/completions", None
        else:
            size = self._pixel_size(aspect_ratio)
            payload = {"model": model, "prompt": prompt, "n": 1, "size": size}
            if model.startswith("gpt-image"):
                payload["quality"] = quality if quality in {"auto", "low", "medium", "high"} else "auto"
            elif model.startswith("dall-e-3"):
                payload["size"] = "1792x1024" if ":" in aspect_ratio and size == "1536x1024" else "1024x1792" if size == "1024x1536" else "1024x1024"
            if self.connection.provider == "dashscope":
                # DashScope's OpenAI-compatible image editing is a JSON extension
                # on generations, not the OpenAI multipart edits endpoint.
                if references:
                    payload["image"] = references[0] if len(references) == 1 else references
                path, form = "images/generations", None
            elif self.connection.provider == "siliconflow":
                sizes = {"1536x1024": "1024x576", "1024x1536": "576x1024", "1024x1024": "512x512" if "flux.2" in model.lower() else "1024x1024"}
                payload = {"model": model, "prompt": prompt, "image_size": sizes[size], "output_format": "png"}
                if references:
                    if len(references) > 1:
                        raise ProviderRequestError("当前硅基流动适配仅支持单张参考图，请减少参考图片数量。", stage="input")
                    payload["image"] = references[0]
                path, form = "images/generations", None
            else:
                form = await self._image_form(model, prompt, size, references, quality) if references else None
                path = "images/edits" if references else "images/generations"
        self.usage["image_requests"] += 1
        # Intentionally no POST retry: timeouts can happen after a paid request succeeds.
        body = await self._request_json("POST", path, payload=payload if form is None else None, form=form)
        self._record_tokens(body)
        values = self._image_values(body, chat)
        if not values:
            raise ProviderRequestError("图像接口没有返回图片，请检查模型及 images/chat/gemini 协议是否匹配。")
        results = []
        for value in values:
            if value.startswith(("http://", "https://", "data:")):
                data = await self._download_bytes(value, max_attempts=3, retry_delay=retry_delay)
                results.append(base64.b64encode(data).decode("ascii"))
            else:
                try:
                    if not base64.b64decode(value, validate=True):
                        raise ValueError
                except (ValueError, binascii.Error) as exc:
                    raise ProviderRequestError("图像接口返回的 base64 数据无效。") from exc
                results.append(value)
        return results
