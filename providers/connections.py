"""Independent, serializable connection settings for text and image models."""

from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


class ConnectionConfigurationError(ValueError):
    """A connection cannot perform the requested operation."""


PLATFORM_PRESETS: dict[str, dict[str, Any]] = {
    "newapi": {"label": "Omgt / New API", "base_url": "https://newapi.omgteam.me/v1"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1"},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "image_api": "chat"},
    "siliconflow": {"label": "硅基流动 SiliconFlow", "base_url": "https://api.siliconflow.cn/v1"},
    "dashscope": {"label": "阿里云百炼 DashScope", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "deepseek": {"label": "DeepSeek（文本）", "base_url": "https://api.deepseek.com/v1", "supports_images": False},
    "gemini": {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com", "text_model": "gemini-3.8-flash", "image_model": "gemini-nano-banana-2.1", "image_api": "gemini"},
    "evolink": {"label": "Evolink", "base_url": "https://api.evolink.ai", "image_api": "evolink"},
    "custom": {"label": "自定义 OpenAI 兼容平台", "base_url": ""},
}
for _preset in PLATFORM_PRESETS.values():
    for _key, _default in {"text_model": "", "image_model": "", "image_api": "images", "supports_text": True, "supports_images": True}.items():
        _preset.setdefault(_key, _default)


def normalize_base_url(url: str, provider: str = "openai", image_api: str = "images") -> str:
    """Accept a service root, API root, or copied endpoint without doubling /v1."""
    value = (url or PLATFORM_PRESETS.get(provider, {}).get("base_url", "")).strip().rstrip("/")
    if not value:
        raise ConnectionConfigurationError("请填写 API Base URL。")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ConnectionConfigurationError("API Base URL 必须是 http(s) 地址，不能包含用户名或密码。")
    if parsed.query or parsed.fragment:
        raise ConnectionConfigurationError("API Base URL 不能包含查询参数或片段。")
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/images/generations", "/images/edits", "/models"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    if provider == "evolink" or image_api == "evolink":
        if path.endswith("/v1"):
            path = path[:-3]
    elif provider != "gemini" and image_api != "gemini" and not path:
        path = "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True)
class ModelConnection:
    provider: str = "openai"
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    image_api: Literal["images", "chat", "gemini", "evolink"] = "images"

    def validate(self, purpose: str = "text", require_model: bool = True) -> "ModelConnection":
        if purpose not in {"text", "image"}:
            raise ConnectionConfigurationError("连接用途必须是 text 或 image。")
        if self.image_api not in {"images", "chat", "gemini", "evolink"}:
            raise ConnectionConfigurationError("不支持的图像 API 协议。")
        normalize_base_url(self.base_url, self.provider, self.image_api)
        if not self.api_key.strip():
            raise ConnectionConfigurationError("请填写此连接自己的 API Key。")
        if require_model and not self.model.strip():
            raise ConnectionConfigurationError("请选择或填写模型名称。")
        preset = PLATFORM_PRESETS.get(self.provider, {})
        if purpose == "image" and not preset.get("supports_images", True):
            raise ConnectionConfigurationError(f"{preset.get('label', self.provider)} 仅支持文本模型，请另外配置图像平台。")
        return self

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "ModelConnection":
        data = values or {}
        provider = str(data.get("provider", data.get("platform", "openai")))
        preset = PLATFORM_PRESETS.get(provider, {})
        return cls(provider=provider, base_url=str(data.get("base_url", preset.get("base_url", "")) or ""),
                   api_key=str(data.get("api_key", "") or ""), model=str(data.get("model", data.get("model_name", "")) or ""),
                   image_api=str(data.get("image_api", preset.get("image_api", "images"))))

    def to_public_dict(self) -> dict[str, str]:
        """Safe for result JSON, logs, and downloads; never includes API credentials."""
        return {"provider": self.provider, "base_url": normalize_base_url(self.base_url, self.provider, self.image_api),
                "model": self.model, "image_api": self.image_api}
