import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from providers.connections import ModelConnection
from utils import jobs
from utils.model_session import GenerationError, ImageBudgetExceeded, call_image_async, call_text_async, model_session


class FakeRuntime:
    def __init__(self, text, image):
        self.text_connection, self.image_connection = text, image
        self.usage = {"total_tokens": 11}
        self.closed = False
    async def __aenter__(self): return self
    async def __aexit__(self, *args): self.closed = True
    async def generate_text(self, **kwargs):
        await asyncio.sleep(0)
        return [self.text_connection.api_key]
    async def generate_image(self, **kwargs): return ["generated-image"]


@pytest.mark.asyncio
async def test_concurrent_jobs_use_distinct_text_keys_and_image_budget():
    from providers import runtime
    with patch.object(runtime, "ModelRuntime", FakeRuntime):
        async def run(key):
            text = ModelConnection(api_key=key, model="text")
            image = ModelConnection(api_key="image-"+key, model="image")
            async with model_session(text, image, max_image_calls=1) as bound:
                assert await call_text_async("ignored", [], {}) == [key]
                await call_image_async("ignored", "prompt", {})
                with pytest.raises(ImageBudgetExceeded):
                    await call_image_async("ignored", "second", {})
                assert bound.image_requests == 1
        await asyncio.gather(run("session-a"), run("session-b"))


@pytest.mark.asyncio
async def test_error_sentinel_is_an_explicit_failure():
    from providers import runtime
    with patch.object(runtime, "ModelRuntime", FakeRuntime), patch.object(FakeRuntime, "generate_text", new=AsyncMock(return_value=["Error"])):
        connection = ModelConnection(api_key="key", model="model")
        async with model_session(connection, connection):
            with pytest.raises(GenerationError):
                await call_text_async("model", [], {})


def test_cancellation_retains_completed_results_and_writes_history(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RESULTS_DIR", tmp_path)
    ready = __import__("threading").Event()
    async def runner(job):
        job.add_result({"candidate_id": 0, "status": "completed"})
        ready.set()
        await asyncio.sleep(30)
    job = jobs.BatchJob([{"candidate_id": 0}, {"candidate_id": 1}], {"max_image_calls": 2}).start(runner)
    assert ready.wait(5)
    job.cancel()
    job._thread.join(5)
    assert not job.is_alive()
    assert job.snapshot()["status"] == "cancelled"
    saved = json.loads(job.path.read_text(encoding="utf-8"))
    assert saved["results"] == [{"candidate_id": 0, "status": "completed"}]


def test_retry_keeps_prior_successes(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RESULTS_DIR", tmp_path)
    async def runner(job): job.add_result({"candidate_id": 1, "status": "completed"})
    job = jobs.BatchJob([{"candidate_id": 1}], {}, [{"candidate_id": 0, "status": "completed"}]).start(runner)
    job._thread.join(5)
    assert [item["candidate_id"] for item in job.snapshot()["results"]] == [0, 1]
    assert job.snapshot()["status"] == "completed"


@pytest.mark.asyncio
async def test_evolink_long_prompt_uses_text_connection_before_single_image_request():
    from providers import runtime
    from utils.model_session import EVOLINK_NANO_PROMPT_BYTES
    text = ModelConnection(provider="newapi", api_key="text-key", model="vision-text")
    image = ModelConnection(provider="evolink", api_key="image-key", model="gemini-nano-banana-2.1")
    short = "白底中文图：输入→共享编码器→九视图特征均值→分类。"
    compress = AsyncMock(return_value=[short])
    draw = AsyncMock(return_value=["generated-image"])
    with (patch.object(runtime, "ModelRuntime", FakeRuntime),
          patch.object(FakeRuntime, "generate_text", compress),
          patch.object(FakeRuntime, "generate_image", draw)):
        async with model_session(text, image) as bound:
            await call_image_async("ignored-model", "图" * 4749,
                                   {"aspect_ratio": "3:2", "quality": "4K"})
            assert bound.text_requests == 1
            assert bound.image_requests == 1
    assert compress.call_args.kwargs["model_name"] == "vision-text"
    assert draw.await_count == 1
    submitted = draw.call_args.kwargs
    assert submitted["prompt"] == short
    assert len(submitted["prompt"].encode("utf-8")) <= EVOLINK_NANO_PROMPT_BYTES
    assert submitted["aspect_ratio"] == "3:2" and submitted["quality"] == "4K"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, expected_text_calls", [("图" * 4749, 2), ("CANNOT_COMPACT", 1)])
async def test_unsuccessful_compaction_never_submits_or_counts_an_image(answer, expected_text_calls):
    from providers import runtime
    text = ModelConnection(provider="newapi", api_key="text-key", model="text")
    image = ModelConnection(provider="evolink", api_key="image-key", model="gemini-nano-banana-2.1")
    compress = AsyncMock(return_value=[answer])
    draw = AsyncMock(return_value=["must-not-be-called"])
    with (patch.object(runtime, "ModelRuntime", FakeRuntime),
          patch.object(FakeRuntime, "generate_text", compress),
          patch.object(FakeRuntime, "generate_image", draw)):
        async with model_session(text, image) as bound:
            with pytest.raises(GenerationError, match="未提交生图"):
                await call_image_async("ignored", "图" * 4749, {})
            assert bound.image_requests == 0
            assert bound.text_requests == expected_text_calls
    draw.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, model, prompt", [
    ("evolink", "gemini-nano-banana-2.1", "短提示"),
    ("newapi", "gemini-nano-banana-2.1", "图" * 4749),
    ("evolink", "another-image-model", "图" * 4749),
])
async def test_short_prompts_and_other_channels_are_not_rewritten(provider, model, prompt):
    from providers import runtime
    text = ModelConnection(api_key="text-key", model="text")
    image = ModelConnection(provider=provider, api_key="image-key", model=model)
    compress = AsyncMock()
    draw = AsyncMock(return_value=["generated-image"])
    with (patch.object(runtime, "ModelRuntime", FakeRuntime),
          patch.object(FakeRuntime, "generate_text", compress),
          patch.object(FakeRuntime, "generate_image", draw)):
        async with model_session(text, image):
            await call_image_async("ignored", prompt, {})
    compress.assert_not_awaited()
    assert draw.call_args.kwargs["prompt"] == prompt
