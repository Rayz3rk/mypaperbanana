"""Load independent model connections without rewriting the user's legacy config."""

import os
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from providers.connections import ModelConnection, PLATFORM_PRESETS

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SETTINGS = "connections.local.yaml"


def _yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        raise ValueError(f"{path.name} 的 YAML 格式无效。") from None
    if not isinstance(values, dict):
        raise ValueError(f"{path.name} 必须是 YAML 对象。")
    return values


def load_connections(root: Path = ROOT) -> tuple[ModelConnection, ModelConnection]:
    """Separate env overrides > local settings > new/legacy YAML configuration."""
    legacy = _yaml(root / "configs" / "model_config.yaml")
    local = _yaml(root / "configs" / LOCAL_SETTINGS)
    supplied = local.get("connections", legacy.get("connections", {})) or {}
    evolink = legacy.get("evolink", {}) or {}
    keys = legacy.get("api_keys", {}) or {}
    defaults = legacy.get("defaults", {}) or {}
    for name, value in (("connections", supplied), ("evolink", evolink), ("api_keys", keys), ("defaults", defaults)):
        if not isinstance(value, dict):
            raise ValueError(f"配置中的 {name} 必须是对象。")
    for purpose in ("text", "image"):
        if supplied.get(purpose) is not None and not isinstance(supplied[purpose], dict):
            raise ValueError(f"connections.{purpose} 必须是对象。")
    if os.getenv("EVOLINK_API_KEY") or evolink.get("api_key"):
        provider = "evolink"
        key = os.getenv("EVOLINK_API_KEY") or evolink.get("api_key", "")
        url = os.getenv("EVOLINK_BASE_URL") or evolink.get("base_url", "https://api.evolink.ai")
        text_model = defaults.get("model_name") or "gemini-2.5-flash"
        image_model = defaults.get("image_model_name") or "nano-banana-2-beta"
    elif os.getenv("GOOGLE_API_KEY") or keys.get("google_api_key"):
        provider, key = "gemini", os.getenv("GOOGLE_API_KEY") or keys.get("google_api_key", "")
        url = PLATFORM_PRESETS[provider]["base_url"]
        text_model = defaults.get("model_name") or PLATFORM_PRESETS[provider]["text_model"]
        image_model = defaults.get("image_model_name") or PLATFORM_PRESETS[provider]["image_model"]
        retired = {"gemini-2.5-flash-preview-05-20", "gemini-2.0-flash-preview-image-generation"}
        if text_model in retired:
            text_model = PLATFORM_PRESETS[provider]["text_model"]
        if image_model in retired:
            image_model = PLATFORM_PRESETS[provider]["image_model"]
    elif os.getenv("OPENAI_API_KEY") or keys.get("openai_api_key"):
        provider, key = "openai", os.getenv("OPENAI_API_KEY") or keys.get("openai_api_key", "")
        url = PLATFORM_PRESETS[provider]["base_url"]
        text_model = defaults.get("model_name") or ""
        image_model = defaults.get("image_model_name") or ""
    else:
        provider, key, text_model, image_model = "newapi", "", "", ""
        url = PLATFORM_PRESETS[provider]["base_url"]
    result = []
    for purpose, model in (("text", text_model), ("image", image_model)):
        connection = ModelConnection.from_dict(supplied.get(purpose) or {
            "provider": provider, "base_url": url, "api_key": key, "model": model,
        })
        changes = {}
        for env_name, field_name in (("PROVIDER", "provider"), ("BASE_URL", "base_url"),
                                     ("API_KEY", "api_key"), ("MODEL", "model"), ("API", "image_api")):
            value = os.getenv(f"{purpose.upper()}_{env_name}")
            if value is not None:
                changes[field_name] = value
        # A changed platform must not accidentally inherit another platform's key.
        if changes.get("provider", connection.provider) != connection.provider:
            new_preset = PLATFORM_PRESETS.get(changes["provider"], {})
            changes.setdefault("api_key", "")
            changes.setdefault("base_url", new_preset.get("base_url", ""))
            changes.setdefault("image_api", new_preset.get("image_api", "images"))
            changes.setdefault("model", new_preset.get(f"{purpose}_model", ""))
        result.append(replace(connection, **changes))
    return result[0], result[1]


def save_connections(text: ModelConnection, image: ModelConnection, root: Path = ROOT) -> Path:
    """Called only by the explicit 'save locally, including keys' UI action."""
    target = root / "configs" / LOCAL_SETTINGS
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump({"connections": {"text": asdict(text), "image": asdict(image)}},
                                       allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(target)
    return target
