import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from providers.connections import ModelConnection
from providers.gemini import GeminiProvider
from providers.openai_compatible import OpenAICompatibleProvider, ProviderRequestError, contents_to_messages
from providers.runtime import EvolinkConnectionProvider


IMAGE_BYTES = b"example-image-bytes"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode("ascii")
DATA_URL = "data:image/png;base64," + IMAGE_B64


def make_provider(provider="newapi", image_api="images", model="gateway-image"):
    return OpenAICompatibleProvider(ModelConnection(provider=provider, base_url="https://example.com/v1", api_key="private-key", model=model, image_api=image_api))


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["timeout", "connection", "non_json", "invalid_response"])
async def test_text_retries_transient_failures_without_changing_request(code):
    provider = make_provider(model="text-model")
    success = {"choices": [{"message": {"content": "complete description"}}]}
    request = AsyncMock(side_effect=[ProviderRequestError("private upstream body", code=code), success])
    with patch.object(provider, "_request_json", request):
        assert await provider.generate_text("text-model", [{"type": "text", "text": "plan"}], retry_delay=0) == ["complete description"]
    assert request.await_count == 2
    assert request.await_args_list[0] == request.await_args_list[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,code", [
    (asyncio.TimeoutError("private upstream details"), "timeout"),
    (aiohttp.ClientConnectionError("private upstream details"), "connection"),
])
async def test_transport_failures_are_classified_and_bounded(failure, code):
    provider = make_provider(model="text-model")
    session = MagicMock()
    session.request.side_effect = failure
    with patch.object(provider, "_get_session", AsyncMock(return_value=session)):
        with pytest.raises(ProviderRequestError) as error:
            await provider.generate_text("text-model", [{"type": "text", "text": "plan"}], max_attempts=2, retry_delay=0)
    assert error.value.code == code and session.request.call_count == 2
    assert "private upstream details" not in str(error.value)


@pytest.mark.asyncio
async def test_http_error_payload_is_classified_without_exposing_private_body():
    provider = make_provider(model="text-model")
    response = MagicMock(status=400)
    response.json = AsyncMock(return_value={"error": {"code": "context_length_exceeded", "message": "private-key and source manuscript"}})
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.request.return_value = context
    with patch.object(provider, "_get_session", AsyncMock(return_value=session)):
        with pytest.raises(ProviderRequestError) as error:
            await provider.generate_text("text-model", [{"type": "text", "text": "plan"}], retry_delay=0)
    assert error.value.code == "input_limit" and error.value.status == 400
    assert session.request.call_count == 1
    assert "private-key" not in str(error.value) and "source manuscript" not in error.value.safe_hint


@pytest.mark.asyncio
async def test_empty_text_is_retried_and_usage_includes_failed_response():
    provider = make_provider(model="text-model")
    request = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": None, "reasoning_content": "private reasoning"}}], "usage": {"total_tokens": 90}},
        {"choices": [{"message": {"content": "actual final description"}}], "usage": {"total_tokens": 10}},
    ])
    with patch.object(provider, "_request_json", request):
        result = await provider.generate_text("text-model", [{"type": "text", "text": "plan"}], retry_delay=0)
    assert result == ["actual final description"]
    assert request.await_count == 2 and provider.usage["total_tokens"] == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,code", [("length", "output_limit"), ("content_filter", "content_filter")])
async def test_truncated_or_blocked_text_does_not_retry_identical_request(reason, code):
    provider = make_provider(model="text-model")
    body = {"choices": [{"finish_reason": reason, "message": {"content": "incomplete description", "reasoning_content": "private reasoning"}}]}
    with patch.object(provider, "_request_json", AsyncMock(return_value=body)) as request:
        with pytest.raises(ProviderRequestError) as error:
            await provider.generate_text("text-model", [{"type": "text", "text": "plan"}], retry_delay=0)
    assert error.value.code == code and request.await_count == 1
    assert "private reasoning" not in str(error.value)


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried():
    provider = make_provider(model="text-model")
    with patch.object(provider, "_request_json", AsyncMock(side_effect=ProviderRequestError("secret body", status=401))) as request:
        with pytest.raises(ProviderRequestError):
            await provider.generate_text("text-model", [{"type": "text", "text": "plan"}], retry_delay=0)
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_gemini_retries_empty_response_and_never_uses_thoughts_as_description():
    provider = GeminiProvider(ModelConnection(provider="gemini", api_key="private", model="test-model"))
    body = {"candidates": [{"content": {"parts": [{"thought": True, "text": "private thoughts"}]}}]}
    success = {"candidates": [{"content": {"parts": [{"text": "final description"}]}}]}
    with patch.object(provider, "_request_json", AsyncMock(side_effect=[body, success])) as request:
        assert await provider.generate_text("test-model", [{"type": "text", "text": "plan"}], retry_delay=0) == ["final description"]
    assert request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("body,code", [
    ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "output_limit"),
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "content_filter"),
])
async def test_gemini_output_limit_and_safety_are_classified_without_retry(body, code):
    provider = GeminiProvider(ModelConnection(provider="gemini", api_key="private", model="test-model"))
    with patch.object(provider, "_request_json", AsyncMock(return_value=body)) as request:
        with pytest.raises(ProviderRequestError) as error:
            await provider.generate_text("test-model", [{"type": "text", "text": "plan"}], retry_delay=0)
    assert error.value.code == code and request.await_count == 1


def test_planner_images_and_source_images_both_reach_text_messages():
    result = contents_to_messages([{"type": "text", "text": "plan"}, {"type": "image", "image_base64": IMAGE_B64},
                                   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMAGE_B64}}])
    parts = result[0]["content"]
    assert len([part for part in parts if part["type"] == "image_url"]) == 2
    assert all(part["image_url"]["url"] == DATA_URL for part in parts if part["type"] == "image_url")


@pytest.mark.asyncio
async def test_model_list_only_uses_read_only_get():
    provider = make_provider(model="")
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"data": [{"id": "available-model"}]})) as request:
        assert await provider.list_models() == ["available-model"]
    request.assert_awaited_once_with("GET", "models")


@pytest.mark.asyncio
async def test_newapi_text_request_uses_its_own_key_and_reports_usage():
    provider = make_provider(model="text-model")
    response = {"choices": [{"message": {"content": "a plan"}}], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}
    with patch.object(provider, "_request_json", new=AsyncMock(return_value=response)) as request:
        assert await provider.generate_text("text-model", [{"type": "text", "text": "plan"}]) == ["a plan"]
    assert provider._headers() == {"Authorization": "Bearer private-key"}
    assert request.await_args.kwargs["payload"]["model"] == "text-model"
    assert provider.usage["total_tokens"] == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5", "o3", "o4-mini"])
async def test_openai_reasoning_models_use_completion_token_parameter(model):
    provider = make_provider(provider="openai", model=model)
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"choices": [{"message": {"content": "answer"}}]})) as request:
        await provider.generate_text(model, [{"type": "text", "text": "question"}], max_output_tokens=1234)
    payload = request.await_args.kwargs["payload"]
    assert payload["max_completion_tokens"] == 1234
    assert "max_tokens" not in payload and "temperature" not in payload


@pytest.mark.asyncio
async def test_deepseek_rejects_visual_input_before_any_http_request():
    provider = make_provider(provider="deepseek")
    with patch.object(provider, "_request_json", new=AsyncMock()) as request:
        with pytest.raises(ProviderRequestError, match="不支持图片"):
            await provider.generate_text("deepseek-chat", [{"type": "image", "image_base64": IMAGE_B64}])
    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_images_api_returns_base64_without_url_download():
    provider = make_provider()
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"data": [{"b64_json": IMAGE_B64}]})) as request:
        assert await provider.generate_image("gateway-image", "draw") == [IMAGE_B64]
    assert request.await_args.args == ("POST", "images/generations")


@pytest.mark.asyncio
async def test_openai_edits_include_original_image_as_multipart():
    provider = make_provider()
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"data": [{"b64_json": IMAGE_B64}]})) as request:
        await provider.generate_image("gateway-image", "edit", image_urls=[DATA_URL])
    assert request.await_args.args == ("POST", "images/edits")
    assert request.await_args.kwargs["payload"] is None
    fields = request.await_args.kwargs["form"]._fields
    image_field = next(field for field in fields if field[0]["name"] == "image")
    assert image_field[2] == IMAGE_BYTES


@pytest.mark.asyncio
async def test_dashscope_edits_send_json_image_to_generations():
    provider = make_provider(provider="dashscope")
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"data": [{"b64_json": IMAGE_B64}]})) as request:
        await provider.generate_image("qwen-image-edit", "edit", image_urls=[DATA_URL])
    assert request.await_args.args == ("POST", "images/generations")
    assert request.await_args.kwargs["payload"]["image"] == DATA_URL
    assert request.await_args.kwargs["form"] is None


@pytest.mark.asyncio
async def test_siliconflow_size_and_images_response_are_supported():
    provider = make_provider(provider="siliconflow")
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"images": [{"url": DATA_URL}]})) as request:
        assert await provider.generate_image("FLUX.2-pro", "draw") == [IMAGE_B64]
    payload = request.await_args.kwargs["payload"]
    assert payload["image_size"] == "1024x576" and "size" not in payload


@pytest.mark.asyncio
async def test_image_chat_extracts_multimodal_image_url_and_passes_reference():
    provider = make_provider(provider="openrouter", image_api="chat")
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"choices": [{"message": {"images": [{"image_url": {"url": DATA_URL}}]}}]})) as request:
        assert await provider.generate_image("gateway-image", "edit", image_urls=[DATA_URL]) == [IMAGE_B64]
    payload = request.await_args.kwargs["payload"]
    assert payload["modalities"] == ["image", "text"]
    assert payload["messages"][0]["content"][1]["image_url"]["url"] == DATA_URL


@pytest.mark.asyncio
async def test_uncertain_image_post_is_not_repeated_even_if_caller_allows_attempts():
    provider = make_provider()
    with patch.object(provider, "_request_json", new=AsyncMock(side_effect=ProviderRequestError("timeout"))) as request:
        with pytest.raises(ProviderRequestError):
            await provider.generate_image("gateway-image", "draw", max_attempts=5, retry_delay=0)
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_download_retries_are_independent_of_single_post_budget():
    provider = make_provider()
    failed = MagicMock()
    failed.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
    failed.__aexit__ = AsyncMock(return_value=False)
    success = MagicMock()
    success.__aenter__ = AsyncMock(return_value=success)
    success.__aexit__ = AsyncMock(return_value=False)
    success.raise_for_status = MagicMock()
    success.read = AsyncMock(return_value=IMAGE_BYTES)
    session = MagicMock()
    session.get.side_effect = [failed, failed, success]
    with patch.object(provider, "_request_json", new=AsyncMock(return_value={"data": [{"url": "https://cdn.example.com/result.png"}]})) as request, patch.object(provider, "_get_session", new=AsyncMock(return_value=session)):
        assert await provider.generate_image("gateway-image", "draw", max_attempts=1, retry_delay=0) == [IMAGE_B64]
    assert session.get.call_count == 3
    assert request.await_count == 1
    assert "headers" not in session.get.call_args.kwargs


@pytest.mark.asyncio
async def test_native_gemini_uses_google_key_and_preserves_reference_picture():
    provider = GeminiProvider(ModelConnection(provider="newapi", base_url="https://example.com", api_key="gemini-key", model="gemini-image", image_api="gemini"))
    body = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": IMAGE_B64}}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6}}
    with patch.object(provider, "_request_json", new=AsyncMock(return_value=body)) as request:
        assert await provider.generate_image("gemini-image", "edit", image_urls=[DATA_URL]) == [IMAGE_B64]
    assert provider.base_url == "https://example.com/v1beta"
    assert provider._headers() == {"x-goog-api-key": "gemini-key"}
    assert request.await_args.kwargs["payload"]["contents"][0]["parts"][1]["inlineData"]["data"] == IMAGE_B64
    assert provider.usage["total_tokens"] == 6


@pytest.mark.asyncio
async def test_evolink_data_url_is_uploaded_and_generation_attempts_are_one():
    provider = EvolinkConnectionProvider(ModelConnection(provider="evolink", api_key="evolink-key", model="image"))
    with patch.object(provider, "upload_image_base64", new=AsyncMock(return_value="https://files.example.com/reference.png")) as upload, patch("providers.evolink.EvolinkProvider.generate_image", new=AsyncMock(return_value=[IMAGE_B64])) as generate:
        assert await provider.generate_image("image", "edit", image_urls=[DATA_URL], max_attempts=5) == [IMAGE_B64]
    upload.assert_awaited_once_with(IMAGE_B64, "image/png")
    assert generate.await_args.kwargs["max_attempts"] == 1
    assert generate.await_args.kwargs["image_urls"] == ["https://files.example.com/reference.png"]
