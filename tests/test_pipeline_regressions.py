"""Offline regressions for failure isolation, retrieval and figure outputs."""

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.visualizer_agent import VisualizerAgent
from providers.openai_compatible import ProviderRequestError
from utils import generation_utils
from utils.config import ExpConfig
from utils.output_options import image_media_type, render_plot
from utils.paperviz_processor import PaperVizProcessor


def png():
    stream = BytesIO()
    Image.new("RGBA", (8, 8), (10, 20, 30, 120)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def config(tmp_path, mode="demo_planner_critic", task="diagram", **kwargs):
    return ExpConfig(dataset_name="Example", task_name=task, exp_mode=mode, work_dir=tmp_path,
                     model_name="text-model", image_model_name="image-model", **kwargs)


def processor(cfg, **agents):
    defaults = {name: SimpleNamespace(process=AsyncMock(side_effect=lambda data, **kwargs: data))
                for name in ("vanilla_agent", "planner_agent", "visualizer_agent", "stylist_agent",
                             "critic_agent", "retriever_agent", "polish_agent")}
    defaults.update(agents)
    return PaperVizProcessor(cfg, **defaults)


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["diagram", "plot"])
@pytest.mark.parametrize("error", [
    ProviderRequestError("private server body", code="timeout"),
    ProviderRequestError("private server body", code="empty_text"),
    ProviderRequestError("private server body", code="output_limit"),
    ProviderRequestError("private server body", status=503),
])
async def test_stylist_failure_preserves_plan_and_continues_critic_without_extra_images(tmp_path, task, error):
    cfg = config(tmp_path, "demo_full", task=task, keep_intermediate_images=True, max_critic_rounds=1)
    plan = "A complete scientifically accurate plan with labels and connections."
    async def planning(data):
        data[f"target_{task}_desc0"] = plan
        return data
    async def draw(data):
        assert f"target_{task}_stylist_desc0" not in data
        data[f"target_{task}_desc0_base64_jpg"] = png()
        return data
    async def critique(data, source):
        assert source == "planner"
        data[f"target_{task}_critic_suggestions0"] = "No changes needed."
        return data
    drawing = AsyncMock(side_effect=draw)
    proc = processor(cfg, planner_agent=SimpleNamespace(process=planning),
                     stylist_agent=SimpleNamespace(process=AsyncMock(side_effect=error)),
                     visualizer_agent=SimpleNamespace(process=drawing), critic_agent=SimpleNamespace(process=critique))
    result = [r async for r in proc.process_queries_batch([{"content": "paper", "visual_intent": "caption"}], do_eval=False)][0]
    assert result["status"] == "completed"
    assert result["stylist_status"] == "fallback" and result[f"target_{task}_desc0"] == plan
    assert "未应用风格优化" in result["stylist_warning"] and "private" not in str(result)
    assert result["eval_image_field"] == f"target_{task}_desc0_base64_jpg"
    assert drawing.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    ProviderRequestError("private secret", status=401),
    ProviderRequestError("private secret", status=403),
    ProviderRequestError("private secret", status=400),
    ProviderRequestError("private secret", code="content_filter"),
])
async def test_stylist_auth_configuration_and_safety_errors_stop_before_image(tmp_path, error):
    cfg = config(tmp_path, "demo_full")
    async def planning(data):
        data["target_diagram_desc0"] = "valid plan"
        return data
    drawing = AsyncMock()
    proc = processor(cfg, planner_agent=SimpleNamespace(process=planning),
                     stylist_agent=SimpleNamespace(process=AsyncMock(side_effect=error)),
                     visualizer_agent=SimpleNamespace(process=drawing))
    result = [r async for r in proc.process_queries_batch([{}], do_eval=False)][0]
    assert result["status"] == "failed" and result["failed_stage"] == "stylist"
    assert result["error_code"] == error.code and "private" not in str(result)
    drawing.assert_not_awaited()


@pytest.mark.asyncio
async def test_stylist_fallback_never_draws_from_invalid_plan(tmp_path):
    proc = processor(config(tmp_path, "demo_full"),
        stylist_agent=SimpleNamespace(process=AsyncMock(side_effect=ProviderRequestError("timeout", code="timeout"))))
    result = [r async for r in proc.process_queries_batch([{"target_diagram_desc0": "Error"}], do_eval=False)][0]
    assert result["status"] == "failed"
    proc.visualizer_agent.process.assert_not_awaited()


@pytest.mark.asyncio
async def test_stylist_cancellation_does_not_trigger_fallback(tmp_path):
    proc = processor(config(tmp_path, "demo_full"),
        stylist_agent=SimpleNamespace(process=AsyncMock(side_effect=asyncio.CancelledError)))
    with pytest.raises(asyncio.CancelledError):
        await proc.process_single_query({"target_diagram_desc0": "valid plan"}, do_eval=False)
    proc.visualizer_agent.process.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_candidate_does_not_discard_other_results_or_expose_error(tmp_path):
    async def generate(data):
        if data["candidate_id"] == 0:
            raise ValueError("secret-api-key and source manuscript")
        data["vanilla_diagram_base64_jpg"] = png()
        return data
    proc = processor(config(tmp_path, "vanilla"), vanilla_agent=SimpleNamespace(process=generate))
    results = [result async for result in proc.process_queries_batch(
        [{"candidate_id": index} for index in range(3)], do_eval=False)]
    assert len(results) == 3
    assert sum(item["status"] == "completed" for item in results) == 2
    failed = next(item for item in results if item["status"] == "failed")
    assert "secret" not in failed["error"]
    assert failed["failed_stage"] == "visualizer"


@pytest.mark.asyncio
async def test_closing_batch_generator_cancels_and_collects_other_tasks(tmp_path):
    second_started, second_cancelled = asyncio.Event(), asyncio.Event()
    proc = processor(config(tmp_path))
    async def run(data, do_eval=False):
        if data["candidate_id"] == 0:
            await second_started.wait()
            return dict(data, status="completed")
        second_started.set()
        try:
            await asyncio.sleep(60)
        finally:
            second_cancelled.set()
    proc.process_single_query = run
    batch = proc.process_queries_batch([{"candidate_id": 0}, {"candidate_id": 1}], max_concurrent=2)
    assert (await anext(batch))["candidate_id"] == 0
    await batch.aclose()
    assert second_cancelled.is_set()


@pytest.mark.asyncio
async def test_identical_candidates_share_one_retrieval(tmp_path):
    cfg = config(tmp_path, "dev_planner")
    async def retrieve(data, **kwargs):
        await asyncio.sleep(0)
        data.update(top10_references=["a"], retrieved_examples=[], retrieval_status="completed", retrieval_count=1)
        return data
    retrieval = AsyncMock(side_effect=retrieve)
    async def draw(data):
        data["target_diagram_desc0_base64_jpg"] = png()
        return data
    proc = processor(cfg, retriever_agent=SimpleNamespace(process=retrieval),
                     visualizer_agent=SimpleNamespace(process=draw))
    inputs = [dict(candidate_id=index, content="same", visual_intent="same") for index in range(3)]
    results = [result async for result in proc.process_queries_batch(inputs, do_eval=False)]
    assert retrieval.await_count == 1
    assert sum(item["retrieval_cache_hit"] for item in results) == 2
    assert all(item["status"] == "completed" for item in results)


@pytest.mark.asyncio
async def test_rounds_fall_back_to_config_and_keep_last_success_on_failure(tmp_path):
    cfg = config(tmp_path, max_critic_rounds=2)
    async def plan(data):
        data["target_diagram_desc0"] = "initial description"
        return data
    async def critique(data, **kwargs):
        index = data["current_critic_round"]
        data[f"target_diagram_critic_desc{index}"] = f"revision {index}"
        data[f"target_diagram_critic_suggestions{index}"] = "Increase legibility"
        return data
    async def visualize(data):
        index = data.get("current_critic_round")
        if index == 1:
            raise RuntimeError("upstream private failure")
        key = "target_diagram_desc0" if index is None else f"target_diagram_critic_desc{index}"
        data[key+"_base64_jpg"] = png()
        return data
    critic = AsyncMock(side_effect=critique)
    proc = processor(cfg, planner_agent=SimpleNamespace(process=plan),
                     visualizer_agent=SimpleNamespace(process=visualize),
                     critic_agent=SimpleNamespace(process=critic))
    results = [item async for item in proc.process_queries_batch(
        [{"content": "source", "visual_intent": "caption"}], do_eval=False)]
    assert critic.await_count == 2
    assert results[0]["status"] == "failed"
    assert results[0]["eval_image_field"] == "target_diagram_critic_desc0_base64_jpg"
    assert results[0][results[0]["eval_image_field"]]


@pytest.mark.asyncio
async def test_one_configured_round_does_not_revert_to_three(tmp_path):
    cfg = config(tmp_path, max_critic_rounds=1)
    async def plan(data):
        data["target_diagram_desc0"] = "initial"
        return data
    async def critique(data, **kwargs):
        index = data["current_critic_round"]
        data[f"target_diagram_critic_desc{index}"] = "revision"
        data[f"target_diagram_critic_suggestions{index}"] = "Larger labels"
        return data
    async def visualize(data):
        index = data.get("current_critic_round")
        key = "target_diagram_desc0" if index is None else f"target_diagram_critic_desc{index}"
        data[key+"_base64_jpg"] = png()
        return data
    critic = AsyncMock(side_effect=critique)
    proc = processor(cfg, planner_agent=SimpleNamespace(process=plan),
                     visualizer_agent=SimpleNamespace(process=visualize),
                     critic_agent=SimpleNamespace(process=critic))
    results = [item async for item in proc.process_queries_batch(
        [{"content": "source", "visual_intent": "caption"}], do_eval=False)]
    assert results[0]["status"] == "completed"
    assert critic.await_count == 1


@pytest.mark.asyncio
async def test_empty_reference_library_never_requests_a_model(tmp_path, monkeypatch):
    path = tmp_path / "data" / "Example" / "diagram"
    path.mkdir(parents=True)
    (path / "ref.json").write_text("[]", encoding="utf-8")
    call = AsyncMock()
    monkeypatch.setattr(generation_utils, "call_text_async", call)
    result = await RetrieverAgent(exp_config=config(tmp_path)).process({"content": "paper", "visual_intent": "caption"})
    call.assert_not_awaited()
    assert result["retrieval_status"] == "empty_or_missing"
    assert result["retrieval_count"] == 0


@pytest.mark.asyncio
async def test_failed_planner_never_starts_paid_image_generation(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    text = AsyncMock(return_value=["Error"])
    image = AsyncMock(return_value=[png()])
    monkeypatch.setattr(generation_utils, "call_text_async", text)
    monkeypatch.setattr(generation_utils, "call_image_async", image)
    proc = processor(cfg, retriever_agent=RetrieverAgent(exp_config=cfg),
                     planner_agent=PlannerAgent(exp_config=cfg), visualizer_agent=VisualizerAgent(exp_config=cfg))
    results = [item async for item in proc.process_queries_batch(
        [{"content": "paper", "visual_intent": "caption"}], do_eval=False)]
    assert results[0]["status"] == "failed"
    image.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_critic_response_cannot_pass_review(tmp_path, monkeypatch):
    text = AsyncMock(return_value=["not a JSON review"])
    monkeypatch.setattr(generation_utils, "call_text_async", text)
    agent = CriticAgent(exp_config=config(tmp_path))
    with pytest.raises(ValueError, match="不能视为审核通过"):
        await agent.process({"content": "source", "visual_intent": "figure", "target_diagram_desc0": "description",
                             "target_diagram_desc0_base64_jpg": png()}, source="planner")
    assert text.await_count == 2


@pytest.mark.asyncio
async def test_fifth_round_uses_image_connection_and_preserves_png(tmp_path, monkeypatch):
    image = AsyncMock(return_value=[png()])
    monkeypatch.setattr(generation_utils, "call_image_async", image)
    agent = VisualizerAgent(exp_config=config(tmp_path, output_language="bilingual", glossary="编码器 = Encoder"))
    result = await agent.process({"current_critic_round": 4, "target_diagram_critic_desc4": "description"})
    image.assert_awaited_once()
    assert "编码器 = Encoder" in image.call_args.kwargs["prompt"]
    key = "target_diagram_critic_desc4_base64_png"
    assert image_media_type(result[key]) == "image/png"
    with Image.open(BytesIO(base64.b64decode(result[key]))) as output:
        assert output.mode == "RGBA"


@pytest.mark.asyncio
async def test_full_mode_does_not_generate_unused_planner_image(tmp_path, monkeypatch):
    image = AsyncMock(return_value=[png()])
    monkeypatch.setattr(generation_utils, "call_image_async", image)
    agent = VisualizerAgent(exp_config=config(tmp_path, "demo_full"))
    result = await agent.process({"target_diagram_desc0": "plan", "target_diagram_stylist_desc0": "styled"})
    assert image.await_count == 1
    assert "target_diagram_desc0_base64_jpg" not in result
    assert result["target_diagram_stylist_desc0_base64_png"]


@pytest.mark.asyncio
async def test_plot_code_is_returned_without_implicit_execution_or_critic(tmp_path, monkeypatch):
    cfg = config(tmp_path, task="plot")
    text = AsyncMock(return_value=["import matplotlib.pyplot as plt\nplt.plot([1,2],[2,3])"])
    monkeypatch.setattr(generation_utils, "call_text_async", text)
    critic = AsyncMock()
    proc = processor(cfg, visualizer_agent=VisualizerAgent(exp_config=cfg),
                     critic_agent=SimpleNamespace(process=critic))
    async def plan(data):
        data["target_plot_desc0"] = "Data points (1,2) and (2,3)"
        return data
    proc.planner_agent.process = plan
    result = await proc.process_single_query({"content": "raw", "visual_intent": "line"}, do_eval=False)
    assert result["status"] == "code_only"
    assert result[result["code_field"]].startswith("import matplotlib")
    critic.assert_not_awaited()


@pytest.mark.asyncio
async def test_opt_in_plot_export_creates_png_svg_and_pdf(tmp_path):
    cfg = config(tmp_path, task="plot", allow_plot_code_execution=True, plot_timeout_seconds=30)
    formats = await render_plot("import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])", cfg)
    assert base64.b64decode(formats["png"]).startswith(b"\x89PNG")
    assert b"<svg" in base64.b64decode(formats["svg"])
    assert base64.b64decode(formats["pdf"]).startswith(b"%PDF")
