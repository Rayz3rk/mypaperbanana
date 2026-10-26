"""Exercise real HTTP transport against a local fake gateway, never a paid API."""

import base64
from io import BytesIO

import pytest
import pytest_asyncio
from aiohttp import web
from PIL import Image

from providers.connections import ModelConnection
from utils.model_session import call_image_async, call_text_async, model_session


@pytest_asyncio.fixture
async def gateway(request):
    stream = BytesIO()
    Image.new("RGBA", (32, 18), (0, 90, 180, 180)).save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    calls = []
    async def text(request):
        payload = await request.json()
        calls.append(("text", request.headers.get("Authorization"), payload))
        prompt = str(payload["messages"])
        if getattr(request_info, "param", {}).get("empty_first") and len(calls) == 1:
            return web.json_response({"choices": [{"message": {"content": None}}], "usage": {"total_tokens": 7}})
        if getattr(request_info, "param", {}).get("style_output_limit") and "Style Guidelines:" in prompt:
            return web.json_response({"choices": [{"finish_reason": "length", "message": {"content": None, "reasoning_content": "not a figure description"}}],
                                      "usage": {"prompt_tokens": 100, "completion_tokens": 8192, "total_tokens": 8292}})
        if "critic_suggestions" in prompt:
            content = '{"critic_suggestions":"Make labels larger", "revised_description":"Revised readable encoder diagram"}'
        else:
            content = "Draw an encoder and decoder with labeled arrows."
        return web.json_response({"choices": [{"message": {"content": content}}],
                                  "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
    async def image(request):
        payload = await request.json()
        calls.append(("image", request.headers.get("Authorization"), payload))
        return web.json_response({"data": [{"b64_json": encoded}]})
    async def models(request):
        calls.append(("models", request.headers.get("Authorization"), {}))
        return web.json_response({"data": [{"id": "test-vision"}, {"id": "test-image"}]})
    async def edit(request):
        reader = await request.multipart()
        fields = {}
        async for part in reader:
            if part.filename:
                data = await part.read()
                with Image.open(BytesIO(data)) as reference:
                    fields["reference_size"] = list(reference.size)
                    fields["reference_format"] = reference.format
            else:
                fields[part.name] = await part.text()
        calls.append(("edit", request.headers.get("Authorization"), fields))
        return web.json_response({"data": [{"b64_json": encoded}]})
    request_info = request
    app = web.Application()
    app.router.add_post("/text/v1/chat/completions", text)
    app.router.add_post("/image/v1/images/generations", image)
    app.router.add_post("/image/v1/images/edits", edit)
    app.router.add_get("/{purpose}/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", calls
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", [{"empty_first": True}], indirect=True)
async def test_empty_text_retries_over_http_and_ui_usage_counts_both_requests(gateway):
    base, calls = gateway
    connection = ModelConnection(provider="newapi", base_url=base+"/text/v1", api_key="text-key", model="text-model")
    async with model_session(connection, connection) as session:
        assert await call_text_async("text-model", [{"type": "text", "text": "plan"}], {}, retry_delay=0)
        assert session.usage()["text_requests"] == 2
        assert session.usage()["total_tokens"] == 37
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", [{"style_output_limit": True}], indirect=True)
async def test_full_pipeline_recovers_from_style_output_limit_over_real_http(gateway):
    from demo import process_parallel_candidates
    base, calls = gateway
    text = ModelConnection(provider="newapi", base_url=base+"/text/v1", api_key="text-key", model="thinking-text")
    image = ModelConnection(provider="newapi", base_url=base+"/image/v1", api_key="image-key", model="test-image")
    inputs = [{"candidate_id": index, "content": "An encoder with a decoder", "visual_intent": "Model architecture", "max_critic_rounds": 1} for index in range(2)]
    results = await process_parallel_candidates(inputs, exp_mode="demo_full", retrieval_setting="none",
        text_connection=text, image_connection=image, keep_intermediate_images=True, max_image_calls=4)
    assert len(results) == 2 and all(r["status"] == "completed" for r in results)
    assert all(r["stylist_status"] == "fallback" and r["stylist_error_code"] == "output_limit" for r in results)
    assert all(r["eval_image_field"] == "target_diagram_critic_desc0_base64_jpg" for r in results)
    assert len([c for c in calls if c[0] == "image"]) == 4
    assert all(c[1] == "Bearer image-key" for c in calls if c[0] == "image")
    assert all("not a figure description" not in str(c[2]) for c in calls if c[0] == "image")


@pytest.mark.asyncio
async def test_newapi_protocol_with_two_addresses_and_two_keys(gateway):
    base, calls = gateway
    text = ModelConnection(provider="newapi", base_url=base+"/text/v1", api_key="text-only-key", model="test-vision")
    image = ModelConnection(provider="newapi", base_url=base+"/image/v1", api_key="image-only-key", model="test-image")
    async with model_session(text, image, max_image_calls=1) as session:
        assert await call_text_async("text", [{"type": "text", "text": "plan"}], {})
        assert await call_image_async("image", "draw", {})
        assert session.usage()["total_tokens"] == 30
    assert [(kind, key) for kind, key, _ in calls] == [("text", "Bearer text-only-key"), ("image", "Bearer image-only-key")]
    assert calls[0][2]["model"] == "test-vision"
    assert calls[1][2]["model"] == "test-image"


@pytest.mark.asyncio
async def test_real_pipeline_runs_all_five_rounds_with_split_connections(gateway):
    from demo import create_sample_inputs, process_parallel_candidates
    from utils.results import evolution_stages, result_downloads
    base, calls = gateway
    text = ModelConnection(provider="newapi", base_url=base+"/text/v1", api_key="planner-key", model="test-vision")
    image = ModelConnection(provider="newapi", base_url=base+"/image/v1", api_key="renderer-key", model="test-image")
    inputs = create_sample_inputs("编码器将输入映射到表示，再解码。", "编码与解码流程", num_copies=2, max_critic_rounds=5)
    results = await process_parallel_candidates(inputs, retrieval_setting="none", text_connection=text,
        image_connection=image, output_language="bilingual", glossary="编码器 = Encoder", max_concurrent=2, max_image_calls=12)
    assert len(results) == 2
    assert all(item["status"] == "completed" for item in results)
    for result in results:
        assert result["eval_image_field"].startswith("target_diagram_critic_desc4")
        assert evolution_stages(result)[-1]["name"] == "评审第 5 轮"
        assert result_downloads(result)["png"].startswith(b"\x89PNG")
    text_calls = [call for call in calls if call[0] == "text"]
    image_calls = [call for call in calls if call[0] == "image"]
    assert len(image_calls) == 12
    assert all(key == "Bearer planner-key" for _, key, _ in text_calls)
    assert all(key == "Bearer renderer-key" for _, key, _ in image_calls)
    assert "编码器 = Encoder" in str(text_calls[0][2])


@pytest.mark.asyncio
async def test_refinement_uses_real_multipart_http_and_only_image_key(gateway):
    from demo import refine_image_with_nanoviz
    base, calls = gateway
    source = BytesIO()
    Image.new("RGBA", (13, 9), (20, 80, 40, 100)).save(source, format="PNG")
    text = ModelConnection(provider="newapi", base_url=base+"/text/v1", api_key="text-unused", model="test-vision")
    image = ModelConnection(provider="newapi", base_url=base+"/image/v1", api_key="edit-key", model="test-image")
    output, _ = await refine_image_with_nanoviz(source.getvalue(), "Enlarge labels",
        text_connection=text, image_connection=image)
    assert output.startswith(b"\x89PNG")
    assert len(calls) == 1
    kind, key, fields = calls[0]
    assert kind == "edit" and key == "Bearer edit-key"
    assert fields["reference_size"] == [13, 9]
    assert fields["reference_format"] == "PNG"
    assert fields["model"] == "test-image"
