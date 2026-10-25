# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Command-line generation with independently configured model connections."""

import argparse
import asyncio
import copy
import json
import os
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from demo import process_parallel_candidates
from providers.connections import PLATFORM_PRESETS
from utils.results import write_history
from utils.settings import load_connections

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="PaperBanana-CN 学术配图生成")
    parser.add_argument("--input-file", type=Path, help="输入 JSON 对象或对象列表（content + visual_intent/caption）")
    parser.add_argument("--output", type=Path, help="输出结果 JSON 路径")
    parser.add_argument("--dataset_name", default="PaperBananaBench")
    parser.add_argument("--task_name", choices=["diagram", "plot"], default="diagram")
    parser.add_argument("--split_name", default="test")
    parser.add_argument("--exp_mode", choices=["demo_planner_critic", "demo_full", "dev_planner_critic", "dev_full"], default="demo_planner_critic")
    parser.add_argument("--retrieval_setting", choices=["auto", "auto-full", "manual", "random", "none"], default="auto")
    parser.add_argument("--max_critic_rounds", type=int, choices=range(6), default=3)
    parser.add_argument("--model_name", "--text-model", dest="model_name", default="")
    parser.add_argument("--image-model", default="")
    parser.add_argument("--text-provider", choices=list(PLATFORM_PRESETS))
    parser.add_argument("--image-provider", choices=list(PLATFORM_PRESETS))
    parser.add_argument("--text-base-url")
    parser.add_argument("--image-base-url")
    parser.add_argument("--image-api", choices=["images", "chat", "gemini", "evolink"])
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--max-image-calls", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--output-language", choices=["auto", "zh", "en", "bilingual"], default="auto")
    parser.add_argument("--glossary", default="", help="术语表文本；密钥应使用环境变量或本地连接配置")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--image-quality", choices=["1K", "2K", "4K"], default="2K")
    parser.add_argument("--keep-intermediate-images", action="store_true")
    parser.add_argument("--allow-plot-code-execution", action="store_true")
    return parser


def _connection_override(connection, args, purpose):
    changes = {}
    provider = getattr(args, f"{purpose}_provider")
    if provider and provider != connection.provider:
        preset = PLATFORM_PRESETS[provider]
        changes.update(provider=provider, api_key=os.getenv(f"{purpose.upper()}_API_KEY", ""),
                       base_url=preset["base_url"], model=preset[f"{purpose}_model"], image_api=preset["image_api"])
    url = getattr(args, f"{purpose}_base_url")
    if url:
        changes["base_url"] = url
    model = args.model_name if purpose == "text" else args.image_model
    if model:
        changes["model"] = model
    if purpose == "image" and args.image_api:
        changes["image_api"] = args.image_api
    return replace(connection, **changes)


async def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.num_candidates <= 20 or not 1 <= args.max_concurrent <= 10:
        parser.error("候选数量须在 1–20，并发须在 1–10。")
    if args.max_image_calls < 0 or args.max_output_tokens < 1:
        parser.error("请求上限不能为负数，输出 token 上限必须大于 0。")
    source = args.input_file or ROOT / "data" / args.dataset_name / args.task_name / f"{args.split_name}.json"
    if not source.is_file():
        parser.error("输入文件不存在；请提供 --input-file，或先准备数据集。")
    incoming = json.loads(source.read_text(encoding="utf-8"))
    incoming = [incoming] if isinstance(incoming, dict) else incoming
    if not isinstance(incoming, list) or not all(isinstance(item, dict) and "content" in item for item in incoming):
        parser.error("输入必须是带 content 字段的对象或对象列表。")
    inputs = []
    for item in incoming:
        for _ in range(args.num_candidates):
            data = copy.deepcopy(item)
            data.update(candidate_id=len(inputs), max_critic_rounds=args.max_critic_rounds)
            data.setdefault("visual_intent", data.get("caption", ""))
            data.setdefault("filename", f"input_{len(inputs)}")
            data.setdefault("additional_info", {})["rounded_ratio"] = args.aspect_ratio
            inputs.append(data)
    text, image = load_connections()
    text, image = _connection_override(text, args, "text"), _connection_override(image, args, "image")
    text.validate("text")
    if args.task_name == "diagram":
        image.validate("image")
    if text.provider == "deepseek" and (args.max_critic_rounds > 0 or args.retrieval_setting != "none"):
        parser.error("DeepSeek 纯文本模型需关闭检索并设置 --max_critic_rounds 0。")
    output = args.output or ROOT / "results" / "demo" / f"cli_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.json"
    public = {"text_connection": text.to_public_dict(),
              "image_connection": {"provider": image.provider, "base_url": image.base_url, "model": image.model, "image_api": image.image_api},
              "exp_mode": args.exp_mode, "task_name": args.task_name, "retrieval_setting": args.retrieval_setting,
              "max_concurrent": args.max_concurrent, "max_image_calls": args.max_image_calls,
              "max_output_tokens": args.max_output_tokens, "output_language": args.output_language,
              "glossary": args.glossary, "image_quality": args.image_quality,
              "keep_intermediate_images": args.keep_intermediate_images,
              "allow_plot_code_execution": args.allow_plot_code_execution}
    payload = {"version": 2, "status": "running", "created_at": datetime.now().astimezone().isoformat(),
               "settings": public, "inputs": inputs, "results": [], "usage": {}}
    class Journal:
        cancel_event = threading.Event()
        def report_stage(self, candidate_id, stage):
            print(f"候选 {candidate_id}: {stage}")
        def report_usage(self, usage):
            payload["usage"] = usage
        def add_result(self, result):
            payload["results"].append(result)
            write_history(output, payload)
    try:
        await process_parallel_candidates(inputs, text_connection=text, image_connection=image, job=Journal(),
            exp_mode=args.exp_mode, retrieval_setting=args.retrieval_setting, task_name=args.task_name,
            max_concurrent=args.max_concurrent, max_image_calls=args.max_image_calls,
            max_output_tokens=args.max_output_tokens, output_language=args.output_language, glossary=args.glossary,
            image_quality=args.image_quality, keep_intermediate_images=args.keep_intermediate_images,
            allow_plot_code_execution=args.allow_plot_code_execution)
        payload["status"] = "partial" if any(item.get("status") == "failed" for item in payload["results"]) else "completed"
    except asyncio.CancelledError:
        payload["status"] = "cancelled"
        raise
    finally:
        write_history(output, payload)
    print(f"结果已保存：{output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
