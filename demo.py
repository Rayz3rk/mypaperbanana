# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""PaperBanana-CN: independent model connections and recoverable figure jobs."""

import asyncio
import base64
import copy
import hashlib
import json
import re
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from providers.connections import ModelConnection, PLATFORM_PRESETS
from utils import config, generation_utils
from utils.jobs import BatchJob
from utils.results import (evolution_stages, final_image_key, list_history, load_history,
                           normalize_history, png_bytes, result_downloads, results_zip)
from utils.settings import load_connections, save_connections

ROOT = Path(__file__).resolve().parent
IMAGE_PROTOCOLS = {"images": "OpenAI Images", "chat": "聊天接口返图", "gemini": "Gemini 原生", "evolink": "Evolink 异步任务"}
LANGUAGES = {"auto": "跟随输入", "zh": "中文", "en": "英文", "bilingual": "中英双语"}
JOB_STATUSES = {"queued": "等待中", "running": "生成中", "completed": "已完成", "partial": "部分失败",
                "cancelled": "已取消", "failed": "失败"}


def _stage_label(stage):
    labels = {"retriever": "检索参考", "retriever_cached": "复用参考", "planner": "规划",
              "stylist": "风格优化", "stylist_fallback": "风格优化未完成，使用原规划",
              "visualizer": "生成初图", "polish": "精修",
              "code_ready": "绘图代码就绪", "completed": "已完成", "failed": "失败"}
    if stage.endswith("_completed"):
        return _stage_label(stage.removesuffix("_completed")) + "完成"
    if stage.startswith("critic_") and stage[7:].isdigit():
        return f"第 {stage[7:]} 轮评审"
    if stage.startswith("visualizer_") and stage[11:].isdigit():
        return f"第 {stage[11:]} 轮生图"
    return labels.get(stage, stage)


def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8") if isinstance(text, str) else text


def base64_to_image(value):
    if not isinstance(value, str):
        return None
    try:
        return Image.open(BytesIO(base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value)))
    except (ValueError, OSError, TypeError):
        return None


def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=5,
                         max_critic_rounds=3, image_size="2K"):
    base = {"filename": "demo_input", "caption": caption, "content": clean_text(method_content),
            "visual_intent": clean_text(caption), "additional_info": {"rounded_ratio": aspect_ratio},
            "max_critic_rounds": max_critic_rounds, "image_size": image_size}
    return [dict(copy.deepcopy(base), filename=f"demo_input_candidate_{index}", candidate_id=index)
            for index in range(num_copies)]


def _make_processor(exp_config):
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils.paperviz_processor import PaperVizProcessor
    return PaperVizProcessor(exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config), planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config), stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config), retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config))


async def process_parallel_candidates(data_list, exp_mode="demo_planner_critic", retrieval_setting="auto",
        model_name="", image_model_name="", provider="evolink", api_key="", *, text_connection=None,
        image_connection=None, max_concurrent=3, max_image_calls=20, max_output_tokens=8192,
        output_language="auto", glossary="", keep_intermediate_images=False, image_quality="2K",
        task_name="diagram", allow_plot_code_execution=False, plot_timeout_seconds=30, job=None):
    default_text, default_image = load_connections()
    text = text_connection or default_text
    image = image_connection or default_image
    if api_key:  # Compatibility for scripts that called the old demo function.
        preset = PLATFORM_PRESETS.get(provider, {})
        text = ModelConnection.from_dict({"provider": provider, "api_key": api_key,
                    "base_url": preset.get("base_url", ""), "model": model_name})
        image = replace(text, model=image_model_name, image_api=preset.get("image_api", "images"))
    text.validate("text")
    if task_name == "diagram":
        image.validate("image")
    rounds = max((item.get("max_critic_rounds", 3) for item in data_list), default=3)
    exp_config = config.ExpConfig(dataset_name="Demo", task_name=task_name, split_name="demo",
        exp_mode=exp_mode, retrieval_setting=retrieval_setting, model_name=text.model,
        image_model_name=image.model, work_dir=ROOT, max_critic_rounds=rounds,
        output_language=output_language, glossary=glossary, image_quality=image_quality,
        keep_intermediate_images=keep_intermediate_images, max_output_tokens=max_output_tokens,
        max_image_calls=max_image_calls, allow_plot_code_execution=allow_plot_code_execution,
        plot_timeout_seconds=plot_timeout_seconds, progress_callback=job.report_stage if job else None)
    processor = _make_processor(exp_config)
    results = []
    async with generation_utils.model_session(text, image, max_image_calls=max_image_calls,
            max_output_tokens=max_output_tokens, cancel_event=job.cancel_event if job else None,
            on_usage=job.report_usage if job else None):
        async for result in processor.process_queries_batch(data_list,
                max_concurrent=max(1, int(max_concurrent)), do_eval=False):
            results.append(result)
            if job:
                job.add_result(result)
    return results


async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="16:9", image_size="2K",
        api_key="", provider="evolink", *, image_connection=None, text_connection=None, job=None):
    text, image = load_connections()
    text = text_connection or text
    image = image_connection or image
    if api_key:
        image = ModelConnection.from_dict({"provider": provider, "api_key": api_key,
            "model": image.model, "image_api": PLATFORM_PRESETS.get(provider, {}).get("image_api", "images")})
    image.validate("image")
    references = None
    if image_bytes:
        # Preserve transparency and avoid palette/CMYK JPEG failures.
        with Image.open(BytesIO(image_bytes)) as original:
            if original.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
                original = original.convert("RGB")
            buffer = BytesIO()
            original.save(buffer, format="PNG")
        references = ["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")]
    async with generation_utils.model_session(text, image, max_image_calls=1,
            cancel_event=job.cancel_event if job else None, on_usage=job.report_usage if job else None):
        values = await generation_utils.call_image_async(image.model, edit_prompt,
            {"aspect_ratio": aspect_ratio, "quality": image_size, "image_urls": references})
    return png_bytes(values[0]), "图像已生成"


def get_evolution_stages(result, exp_mode):
    return evolution_stages(result, exp_mode)


def display_candidate_result(result, candidate_id, exp_mode, widget_prefix=None):
    widget_prefix = widget_prefix or str(candidate_id)
    st.markdown(f"**候选 {candidate_id + 1}**")
    if result.get("status") == "failed":
        st.error(result.get("error", "此候选失败，可单独重试。"))
    if isinstance(result.get("stylist_warning"), str):
        st.warning(result["stylist_warning"])
    key = final_image_key(result, exp_mode)
    if key:
        image = base64_to_image(result[key])
        if image:
            st.image(image, width="stretch")
            st.caption(f"实际尺寸 {image.width} × {image.height} px")
            try:
                downloads = result_downloads(result, exp_mode)
            except (ValueError, OSError, TypeError):
                downloads = {}
                st.warning("此历史图片数据不完整，无法导出。")
            for extension, data in downloads.items():
                st.download_button(f"下载 {extension.upper()}", data,
                    file_name=f"candidate_{candidate_id + 1}.{extension}",
                    mime={"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[extension],
                    key=f"download_{widget_prefix}_{extension}")
    if result.get("retrieval_status"):
        st.caption(f"参考检索：{result['retrieval_status']} · {result.get('retrieval_count', 0)} 个参考")
    with st.expander("演化过程与描述"):
        for stage in evolution_stages(result, exp_mode):
            st.markdown(f"**{stage['name']}**")
            image = base64_to_image(result[stage["image_key"]])
            if image:
                st.image(image, width="stretch")
            st.text(result.get(stage["desc_key"], ""))
            suggestions = result.get(stage["suggestions_key"])
            if suggestions:
                st.text(str(suggestions))
        if not key:
            for name, value in result.items():
                if name.startswith("target_") and ("code" in name or name.endswith("desc0")) and isinstance(value, str):
                    is_code = name.endswith("_code")
                    match = re.search(r"```(?:python)?\s*(.*?)```", value, re.DOTALL) if is_code else None
                    content = match.group(1).strip() if match else value
                    if is_code:
                        st.code(content, language="python")
                    else:
                        st.text(content)
                    st.download_button("下载绘图代码" if is_code else "下载描述", content,
                                       file_name=f"candidate_{candidate_id + 1}.{'py' if is_code else 'txt'}",
                                       key=f"code_{widget_prefix}_{name}")


def _reset_connection(purpose):
    preset = PLATFORM_PRESETS[st.session_state[f"{purpose}_provider"]]
    for field, value in {"base_url": preset["base_url"], "api_key": "", "model": preset[f"{purpose}_model"],
                         "image_api": preset["image_api"]}.items():
        st.session_state[f"{purpose}_{field}"] = value
    st.session_state.pop(f"{purpose}_models", None)


def _choose_model(purpose):
    chosen = st.session_state.get(f"{purpose}_model_choice")
    if chosen:
        st.session_state[f"{purpose}_model"] = chosen


def _connection_values(purpose):
    return ModelConnection(**{field: st.session_state.get(f"{purpose}_{field}", "images" if field == "image_api" else "")
                              for field in ("provider", "base_url", "api_key", "model", "image_api")})


async def _fetch_models(connection, purpose):
    from providers.runtime import create_connection_provider
    connection.validate(purpose, require_model=False)
    provider = create_connection_provider(connection)
    try:
        return await provider.list_models()
    finally:
        await provider.close()


def _connection_editor(purpose, connection):
    title = "文本模型连接" if purpose == "text" else "图像模型连接"
    st.markdown(f"### {title}")
    for field in ("provider", "base_url", "api_key", "model", "image_api"):
        st.session_state.setdefault(f"{purpose}_{field}", getattr(connection, field))
    platforms = [name for name, preset in PLATFORM_PRESETS.items()
                 if purpose == "text" or preset["supports_images"]]
    current = st.session_state[f"{purpose}_provider"]
    if current not in platforms:
        st.session_state[f"{purpose}_provider"] = "custom"
    st.selectbox("平台", platforms, format_func=lambda value: PLATFORM_PRESETS[value]["label"],
                 key=f"{purpose}_provider", on_change=_reset_connection, args=(purpose,))
    st.text_input("API Base URL", key=f"{purpose}_base_url", help="可填写服务根地址或 /v1 地址；文本与图像可用不同地址。")
    st.text_input("API Key", type="password", key=f"{purpose}_api_key", help="此密钥仅用于这一组连接，不会自动借用另一组密钥。")
    if purpose == "image":
        st.selectbox("图像接口协议", list(IMAGE_PROTOCOLS), format_func=IMAGE_PROTOCOLS.get, key=f"{purpose}_image_api")
    if st.button("读取模型列表 / 检查连接", key=f"{purpose}_check"):
        try:
            with st.spinner("正在读取可用模型…"):
                models = asyncio.run(_fetch_models(_connection_values(purpose), purpose))
            st.session_state[f"{purpose}_models"] = models
            st.success(f"连接成功，读取到 {len(models)} 个模型。未发起生成请求。")
        except Exception as exc:
            from providers.openai_compatible import ProviderRequestError
            message = str(exc) if isinstance(exc, (ValueError, ProviderRequestError)) else type(exc).__name__
            # Avoid displaying credentials accidentally echoed by a third-party server.
            secret = st.session_state.get(f"{purpose}_api_key", "")
            st.error(message.replace(secret, "[隐藏]") if secret else message)
    models = st.session_state.get(f"{purpose}_models", [])
    if models:
        st.selectbox("从列表选择模型", [""] + models, key=f"{purpose}_model_choice",
                     on_change=_choose_model, args=(purpose,))
    st.text_input("模型名称", key=f"{purpose}_model", help="以当前 Key 的模型列表和渠道支持为准，也可手动填写。")
    if purpose == "text":
        st.caption("规划和评审会使用图片输入。纯文本模型请关闭参考检索，并将评审轮次设为 0。")
    else:
        st.caption("New API 默认使用 OpenAI Images；若渠道模型要求聊天返图或 Gemini 原生接口，请切换协议。")
    return _connection_values(purpose)


def _reference_status(task):
    path = ROOT / "data" / "PaperBananaBench" / task / "ref.json"
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return len(values) if isinstance(values, list) else 0
    except (OSError, ValueError):
        return 0


@st.cache_data
def _cjk_fonts():
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei") if name in available]


def _pending_inputs(payload):
    if payload.get("settings", {}).get("workflow") == "refine":
        return []
    completed = {str(result.get("candidate_id")) for result in payload.get("results", [])
                 if result.get("status", "completed") != "failed"}
    return [copy.deepcopy(item) for item in payload.get("inputs", []) if str(item.get("candidate_id")) not in completed]


def _start_batch(inputs, settings, text, image, retained=None):
    job = BatchJob(inputs, settings, retained_results=retained)
    async def runner(current):
        await process_parallel_candidates(inputs, text_connection=text, image_connection=image, job=current,
            **{name: settings[name] for name in ("exp_mode", "retrieval_setting", "max_concurrent", "max_image_calls",
                "max_output_tokens", "output_language", "glossary", "keep_intermediate_images", "image_quality",
                "task_name", "allow_plot_code_execution") if name in settings})
    st.session_state["active_job"] = job.start(runner)
    st.session_state.pop("display_payload", None)


def _render_job(job):
    @st.fragment(run_every=1.0 if job.is_alive() else None)
    def progress():
        payload = job.snapshot()
        done, total = len(payload["results"]), payload["total"]
        status = JOB_STATUSES.get(payload["status"], payload["status"])
        st.progress(min(done / max(total, 1), 1.0), text=f"已返回 {done} / {total} 个候选 · {status}")
        usage = payload["usage"]
        st.caption(f"文本请求 {usage.get('text_requests', 0)} · 生图请求 {usage.get('image_requests', 0)} / "
                   f"{usage.get('max_image_calls', payload['settings'].get('max_image_calls', 0))} · "
                   f"已报告 tokens {usage.get('total_tokens', 0)}")
        if payload["stages"]:
            st.write(" · ".join(f"候选 {int(key) + 1 if str(key).isdigit() else key}：{_stage_label(value)}"
                                for key, value in payload["stages"].items()))
        if job.is_alive():
            if st.button("取消任务并保留已完成结果", key=f"cancel_{job.id}"):
                job.cancel()
                st.info("正在停止后续请求…")
            if payload["results"]:
                with st.expander("已返回的候选（可立即查看和下载）", expanded=True):
                    columns = st.columns(3)
                    for index, result in enumerate(payload["results"]):
                        identity = result.get("candidate_id", index)
                        identity = identity if isinstance(identity, int) and identity >= 0 else index
                        with columns[index % 3]:
                            display_candidate_result(result, identity, payload["settings"].get("exp_mode", ""),
                                                     widget_prefix=f"live_{job.id}_{index}")
        elif st.session_state.get("finished_job_id") != job.id:
            st.session_state["finished_job_id"] = job.id
            st.session_state["display_payload"] = payload
            st.rerun()
    progress()


def _show_results(payload, text, image):
    results = payload.get("results", [])
    settings = payload.get("settings", {})
    if payload.get("error"):
        st.error(payload["error"])
    pending = _pending_inputs(payload)
    if pending and st.button(f"只重试未完成 / 失败的 {len(pending)} 个候选"):
        try:
            text.validate("text")
            if settings.get("task_name", "diagram") == "diagram":
                image.validate("image")
            retained = [result for result in results if result.get("status", "completed") != "failed"]
            _start_batch(pending, settings, text, image, retained)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if not results:
        if payload.get("status") == "cancelled":
            st.info("任务已取消，已停止后续请求。")
        return
    st.subheader("候选结果")
    st.caption(f"{payload.get('created_at', '')} · 状态：{JOB_STATUSES.get(payload.get('status'), '已完成')}")
    columns = st.columns(3)
    for index, result in enumerate(results):
        with columns[index % 3]:
            identity = result.get("candidate_id", index)
            stable_id = identity if isinstance(identity, int) and not isinstance(identity, bool) and identity >= 0 else index
            display_candidate_result(result, stable_id, settings.get("exp_mode", ""),
                                     widget_prefix=f"{payload.get('id', 'history')}_{index}")
    st.download_button("下载完整结果 JSON（不含密钥）", json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name="paperbanana_results.json", mime="application/json")
    try:
        st.download_button("批量下载 PNG / SVG / PDF", results_zip(results, settings.get("exp_mode", "")),
                           file_name="paperbanana_figures.zip", mime="application/zip")
    except (ValueError, OSError, TypeError):
        st.warning("部分历史图片数据无效，请分别下载可用结果。")


def main():
    st.set_page_config(layout="wide", page_title="PaperBanana-CN 学术配图助手", page_icon="🍌")
    st.title("🍌 PaperBanana-CN 学术配图助手")
    st.caption("从论文方法到学术配图 · 文本与图像模型独立配置 · 结果自动保存")
    try:
        default_text, default_image = load_connections()
    except (ValueError, OSError) as exc:
        st.error(f"配置读取失败：{type(exc).__name__}。请检查本地 YAML 格式。")
        return
    tab_generate, tab_refine, tab_connections, tab_history = st.tabs(["生成候选方案", "图片精修 / 文字生图", "模型连接", "历史结果"])
    with tab_connections:
        st.caption("两组连接可使用不同平台、Base URL、模型和 API Key。切换平台会清空该组密钥，避免误发到其他服务商。")
        col1, col2 = st.columns(2)
        with col1:
            text = _connection_editor("text", default_text)
        with col2:
            image = _connection_editor("image", default_image)
        if st.button("保存连接到本机（包含密钥）"):
            try:
                path = save_connections(text, image)
                st.success(f"已保存到 {path.name}。")
            except OSError:
                st.error("保存失败，请检查 configs 目录写入权限。")
        st.caption("密钥默认仅保留在当前会话；点击保存后写入已加入忽略列表的 connections.local.yaml。")
    job = st.session_state.get("active_job")
    running = job is not None and job.is_alive()
    with st.sidebar:
        st.header("生成设置")
        task = st.selectbox("任务类型", ["diagram", "plot"], format_func=lambda item: "学术配图" if item == "diagram" else "统计图（代码绘制）")
        mode = st.selectbox("流水线", ["demo_planner_critic", "demo_full"], format_func=lambda item: "规划 + 评审" if item == "demo_planner_critic" else "规划 + 风格优化 + 评审")
        retrieval = st.selectbox("参考检索", ["auto", "auto-full", "random", "none"],
            format_func=lambda item: {"auto": "自动（图注）", "auto-full": "自动（完整正文，消耗较高）", "random": "随机参考", "none": "关闭检索"}[item])
        reference_count = _reference_status(task)
        st.caption(f"本地参考库：{reference_count} 条；空库会自动跳过，不发起检索请求。")
        language = st.selectbox("图中文字语言", list(LANGUAGES), format_func=LANGUAGES.get)
        glossary = st.text_area("术语表（每行一个术语或中英对应）", height=90, placeholder="注意力机制 = Attention\n编码器 = Encoder")
        candidates = st.number_input("候选数量", min_value=1, max_value=20, value=5)
        concurrency = st.number_input("并发候选数", min_value=1, max_value=10, value=3)
        rounds = st.number_input("最大评审轮次", min_value=0, max_value=5, value=3)
        aspect = st.selectbox("请求宽高比", ["16:9", "21:9", "3:2", "1:1", "9:16"])
        quality = st.selectbox("请求分辨率", ["1K", "2K", "4K"], index=1)
        keep = st.checkbox("额外生成规划阶段中间图", value=False, disabled=mode != "demo_full")
        expected = int(candidates) * (1 + int(rounds) + int(keep and mode == "demo_full"))
        limit = st.number_input("本次生图请求上限", min_value=1, max_value=200, value=20)
        max_tokens = st.number_input("单次文本最大输出 tokens", min_value=512, max_value=32768, value=8192, step=512)
        allow_plot = st.checkbox("允许在独立子进程执行生成的绘图代码", value=False) if task == "plot" else False
        if task == "plot" and language in {"auto", "zh", "bilingual"}:
            fonts = _cjk_fonts()
            if fonts:
                st.caption("可用中文字体：" + "、".join(fonts))
            else:
                st.info("未检测到常用中文字体；绘制中文统计图前请安装 Noto Sans CJK SC 等字体。")
        st.caption(f"当前配置最多约 {expected} 次生图请求（提前停止会减少）。上限按请求次数计算，不是金额保证。")
        st.caption("尺寸能力因平台和模型不同，结果以实际像素为准。统计图执行选项只控制代码执行，不是权限沙箱。")
    settings = {"exp_mode": mode, "retrieval_setting": retrieval, "max_concurrent": int(concurrency),
        "max_image_calls": int(limit), "max_output_tokens": int(max_tokens), "output_language": language,
        "glossary": glossary, "keep_intermediate_images": keep, "image_quality": quality,
        "task_name": task, "allow_plot_code_execution": allow_plot,
        "text_connection": {"provider": text.provider, "base_url": text.base_url, "model": text.model, "image_api": text.image_api},
        "image_connection": {"provider": image.provider, "base_url": image.base_url, "model": image.model, "image_api": image.image_api}}
    with tab_generate:
        method = st.text_area("论文方法章节 / 统计图原始数据", height=230, key="method_content",
                              placeholder="粘贴论文的方法章节；统计图请粘贴 JSON / 表格数值。")
        caption = st.text_area("图注 / 绘图目标", height=90, key="caption")
        if st.button("生成候选方案", type="primary", disabled=running):
            try:
                if not method.strip() or not caption.strip():
                    raise ValueError("请同时填写方法内容和图注。")
                text.validate("text")
                if task == "diagram":
                    image.validate("image")
                if text.provider == "deepseek" and (rounds > 0 or retrieval != "none"):
                    raise ValueError("DeepSeek 为纯文本模型：请关闭检索并将评审轮次设为 0，或选择支持图像输入的文本模型。")
                inputs = create_sample_inputs(method, caption, aspect_ratio=aspect, num_copies=int(candidates),
                                              max_critic_rounds=int(rounds), image_size=quality)
                _start_batch(inputs, settings, text, image)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if job:
            _render_job(job)
        if st.session_state.get("display_payload"):
            _show_results(st.session_state["display_payload"], text, image)
    with tab_refine:
        st.subheader("图片精修，或直接输入描述生成新图")
        uploaded = st.file_uploader("参考图片（可选）", type=["png", "jpg", "jpeg", "webp"], key="refine_upload")
        edit = st.text_area("生成描述 / 修改要求", height=120, key="edit_prompt")
        if uploaded:
            st.image(uploaded, caption="当前参考图片", width=400)
        if st.button("精修 / 生成图片", type="primary", disabled=running):
            try:
                image.validate("image")
                if not edit.strip():
                    raise ValueError("请填写生成描述或修改要求。")
                source = uploaded.getvalue() if uploaded else None
                refine_settings = dict(settings, workflow="refine", source_sha256=hashlib.sha256(source).hexdigest() if source else "",
                                       max_image_calls=1)
                refine_job = BatchJob([{"candidate_id": 0, "content": edit}], refine_settings)
                async def runner(current):
                    current.report_stage(0, "图像生成 / 精修")
                    data, _ = await refine_image_with_nanoviz(source, edit, aspect, quality,
                        text_connection=text, image_connection=image, job=current)
                    encoded = base64.b64encode(data).decode("ascii")
                    current.add_result({"candidate_id": 0, "status": "completed", "target_diagram_desc0": edit,
                        "target_diagram_desc0_base64_png": encoded, "eval_image_field": "target_diagram_desc0_base64_png"})
                st.session_state["active_job"] = refine_job.start(runner)
                st.session_state.pop("display_payload", None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.caption("任务状态和完成结果显示在“生成候选方案”页；每次结果均保存其原始参数，不会被当前设置覆盖。")
    with tab_history:
        st.subheader("本机历史结果")
        histories = list_history()
        chosen = st.selectbox("已保存的任务", histories, format_func=lambda path: path.name) if histories else None
        if st.button("载入历史结果", disabled=not histories or running):
            try:
                st.session_state["display_payload"] = load_history(chosen)
                st.session_state.pop("active_job", None)
                st.success("已载入，请切换到生成候选方案查看、下载或重试。")
            except (ValueError, OSError):
                st.error("无法读取此结果文件，请检查 JSON 内容。")
        imported = st.file_uploader("导入旧版 / 新版结果 JSON", type=["json"], key="history_import")
        if st.button("载入导入文件", disabled=imported is None or running):
            try:
                st.session_state["display_payload"] = normalize_history(json.loads(imported.getvalue()))
                st.session_state.pop("active_job", None)
                st.success("已载入，请切换到生成候选方案查看。")
            except (ValueError, UnicodeError):
                st.error("结果文件格式无效。")
        st.caption("历史保存在本机 results/demo；结果 JSON 包含论文输入和图片，不包含模型密钥。旧版只有结果数组时可查看下载，但缺少任务输入的文件不能自动续跑。")


if __name__ == "__main__":
    main()
