"""Request-scoped connections: text and images never share implicit credentials."""

import asyncio
from typing import Any

from .connections import ModelConnection, normalize_base_url
from .evolink import EvolinkProvider
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider, ProviderRequestError, decode_data_url


class EvolinkConnectionProvider(EvolinkProvider):
    """Compatibility adapter which uploads data URLs and submits images once."""
    def __init__(self, connection: ModelConnection):
        super().__init__(api_key=connection.api_key, base_url=normalize_base_url(connection.base_url, "evolink", "evolink"))
        self.connection = connection
        self.usage = {"text_requests": 0, "image_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def list_models(self) -> list[str]:
        self.connection.validate(require_model=False)
        response = await self._get_json(f"{self.base_url}/v1/models")
        return sorted({item["id"] for item in response.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)})

    async def generate_text(self, *args, **kwargs) -> list[str]:
        self.connection.validate("text", require_model=not bool(kwargs.get("model_name") or args))
        result = await super().generate_text(*args, **kwargs)
        if not result or any(value == "Error" or not value for value in result):
            raise ProviderRequestError("Evolink 文本生成失败，已停止后续生图。")
        return result

    async def generate_image(self, model_name: str, prompt: str, aspect_ratio: str = "16:9", quality: str = "2K",
                             image_urls: list[str] | None = None, max_attempts: int = 1, retry_delay: float = 1,
                             poll_interval: float = 3, error_context: str = "") -> list[str]:
        self.connection.validate("image", require_model=not bool(model_name))
        urls = []
        for url in image_urls or []:
            if url.startswith("data:"):
                data, mime = decode_data_url(url)
                import base64
                uploaded = await self.upload_image_base64(base64.b64encode(data).decode("ascii"), mime)
                if not uploaded:
                    raise ProviderRequestError("参考图片上传失败，未提交图像生成任务。")
                urls.append(uploaded)
            else:
                urls.append(url)
        result = await super().generate_image(model_name=model_name, prompt=prompt, aspect_ratio=aspect_ratio,
                                              quality=quality, image_urls=urls, max_attempts=1,
                                              retry_delay=retry_delay, poll_interval=poll_interval,
                                              error_context=error_context)
        if not result or any(value == "Error" or not value for value in result):
            raise ProviderRequestError("Evolink 图像生成或下载失败。请检查服务商任务记录，不会自动重复提交。")
        return result

    def _record_tokens(self, body: dict[str, Any]) -> None:
        usage = body.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                self.usage[key] += int(value)

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = "image_requests" if "/images/generations" in url else "text_requests"
        self.usage[key] += 1
        response = await super()._post_json(url, payload)
        self._record_tokens(response)
        return response

    async def _get_json(self, url: str) -> dict[str, Any]:
        response = await super()._get_json(url)
        if response.get("status") == "completed":
            self._record_tokens(response)
        return response


def create_connection_provider(connection: ModelConnection):
    if connection.provider == "gemini" or connection.image_api == "gemini":
        return GeminiProvider(connection)
    if connection.provider == "evolink" or connection.image_api == "evolink":
        return EvolinkConnectionProvider(connection)
    return OpenAICompatibleProvider(connection)


class ModelRuntime:
    def __init__(self, text_connection: ModelConnection, image_connection: ModelConnection):
        self.text_connection, self.image_connection = text_connection, image_connection
        self._text_provider = None
        self._image_provider = None

    @property
    def text_provider(self):
        if self._text_provider is None:
            self._text_provider = create_connection_provider(self.text_connection)
        return self._text_provider

    @property
    def image_provider(self):
        if self._image_provider is None:
            self._image_provider = create_connection_provider(self.image_connection)
        return self._image_provider

    async def __aenter__(self) -> "ModelRuntime":
        # Validate when used: model-list requests and plot-only tasks need not
        # have both models configured in advance.
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        providers = [provider for provider in (self._text_provider, self._image_provider) if provider is not None]
        outcomes = await asyncio.gather(*(provider.close() for provider in providers), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                raise outcome

    @property
    def usage(self) -> dict[str, int]:
        keys = {"text_requests", "image_requests", "prompt_tokens", "completion_tokens", "total_tokens"}
        return {key: sum(int(getattr(provider, "usage", {}).get(key, 0)) for provider in (self._text_provider, self._image_provider) if provider is not None) for key in keys}

    async def list_models(self, purpose: str = "text") -> list[str]:
        if purpose not in {"text", "image"}:
            raise ValueError("purpose must be text or image")
        return await (self.text_provider if purpose == "text" else self.image_provider).list_models()

    async def generate_text(self, model_name: str = "", contents: list[dict[str, Any]] | None = None, **kwargs) -> list[str]:
        return await self.text_provider.generate_text(model_name=self.text_connection.model or model_name, contents=contents or [], **kwargs)

    async def generate_image(self, model_name: str = "", prompt: str = "", **kwargs) -> list[str]:
        return await self.image_provider.generate_image(model_name=self.image_connection.model or model_name, prompt=prompt, **kwargs)
