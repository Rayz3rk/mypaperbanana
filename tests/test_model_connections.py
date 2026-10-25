from unittest.mock import AsyncMock, patch

import pytest

from providers.connections import ConnectionConfigurationError, ModelConnection, PLATFORM_PRESETS, normalize_base_url
from providers.runtime import ModelRuntime


@pytest.mark.parametrize("url,expected", [
    ("https://newapi.omgteam.me", "https://newapi.omgteam.me/v1"),
    ("https://newapi.omgteam.me/v1/", "https://newapi.omgteam.me/v1"),
    ("https://newapi.omgteam.me/v1/chat/completions", "https://newapi.omgteam.me/v1"),
    ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
])
def test_normalize_compatible_base_url(url, expected):
    assert normalize_base_url(url) == expected


def test_credentials_are_not_in_repr_or_public_configuration():
    connection = ModelConnection(provider="newapi", api_key="private-credential", model="custom-model")
    assert "private-credential" not in repr(connection)
    assert "api_key" not in connection.to_public_dict()
    assert "private-credential" not in str(connection.to_public_dict())


@pytest.mark.parametrize("url", ["file:///tmp/api", "https://user:password@example.com/v1", "https://example.com/v1?api_key=secret"])
def test_invalid_base_urls_do_not_echo_sensitive_url(url):
    with pytest.raises(ConnectionConfigurationError) as error:
        normalize_base_url(url)
    assert "password" not in str(error.value)
    assert "secret" not in str(error.value)


def test_deepseek_cannot_be_selected_for_image_generation():
    with pytest.raises(ConnectionConfigurationError, match="仅支持文本"):
        ModelConnection(provider="deepseek", api_key="key", model="model").validate("image")


def test_reading_model_list_does_not_require_a_model_to_be_selected():
    ModelConnection(provider="newapi", api_key="key").validate(require_model=False)
    with pytest.raises(ConnectionConfigurationError):
        ModelConnection(provider="newapi", api_key="key").validate()


def test_presets_do_not_invent_gateway_model_names():
    assert PLATFORM_PRESETS["newapi"]["base_url"] == "https://newapi.omgteam.me/v1"
    assert not PLATFORM_PRESETS["newapi"]["text_model"]
    assert not PLATFORM_PRESETS["newapi"]["image_model"]
    assert ModelConnection.from_dict({"provider": "openrouter"}).image_api == "chat"


@pytest.mark.asyncio
async def test_unused_unconfigured_side_does_not_prevent_image_only_workflow():
    missing_text = ModelConnection(provider="custom")
    image = ModelConnection(provider="newapi", api_key="image-key", model="image-model")
    provider = AsyncMock()
    provider.usage = {"image_requests": 1, "total_tokens": 12}
    provider.generate_image.return_value = ["picture"]
    with patch("providers.runtime.create_connection_provider", return_value=provider) as factory:
        async with ModelRuntime(missing_text, image) as runtime:
            assert runtime.usage["total_tokens"] == 0
            assert await runtime.generate_image(prompt="draw") == ["picture"]
            assert runtime.usage["total_tokens"] == 12
        factory.assert_called_once_with(image)
        provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_uses_connection_model_and_distinct_provider_instances():
    text = ModelConnection(provider="newapi", api_key="text-key", model="text-model")
    image = ModelConnection(provider="newapi", api_key="image-key", model="image-model")
    first, second = AsyncMock(), AsyncMock()
    first.usage = {"total_tokens": 7}
    second.usage = {"total_tokens": 3}
    with patch("providers.runtime.create_connection_provider", side_effect=[first, second]):
        async with ModelRuntime(text, image) as runtime:
            await runtime.generate_text(model_name="legacy-model", contents=[{"type": "text", "text": "plan"}])
            await runtime.generate_image(model_name="legacy-image", prompt="draw")
            assert runtime.usage["total_tokens"] == 10
    assert first.generate_text.await_args.kwargs["model_name"] == "text-model"
    assert second.generate_image.await_args.kwargs["model_name"] == "image-model"
    first.close.assert_awaited_once()
    second.close.assert_awaited_once()
