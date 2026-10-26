# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-candidate failure isolation, shared retrieval and cancellable pipelines."""
import asyncio
from copy import deepcopy
import hashlib
import json
from typing import Any, AsyncGenerator, Dict, List

from providers.openai_compatible import ProviderRequestError
from .output_options import reference_root, require_description, resolve_rounds


class PaperVizProcessor:
    def __init__(self, exp_config, vanilla_agent, planner_agent, visualizer_agent,
                 stylist_agent, critic_agent, retriever_agent, polish_agent):
        self.exp_config = exp_config
        self.vanilla_agent = vanilla_agent
        self.planner_agent = planner_agent
        self.visualizer_agent = visualizer_agent
        self.stylist_agent = stylist_agent
        self.critic_agent = critic_agent
        self.retriever_agent = retriever_agent
        self.polish_agent = polish_agent
        self._retrieval_cache = {}
        self._retrieval_locks = {}

    def _progress(self, data: dict, stage: str) -> None:
        data["stage"] = stage
        callback = getattr(self.exp_config, "progress_callback", None)
        if callback:
            callback(data.get("candidate_id", data.get("filename", "")), stage)

    async def _stage(self, data: dict, name: str, agent, **kwargs) -> dict:
        self._progress(data, name)
        result = await agent.process(data, **kwargs)
        self._progress(result, name+"_completed")
        return result

    async def _style(self, data: dict) -> dict:
        try:
            result = await self._stage(data, "stylist", self.stylist_agent)
            result["stylist_status"] = "completed"
            return result
        except ProviderRequestError as exc:
            # Aesthetic refinement is optional; only a valid completed plan may
            # replace it. Authentication, configuration and safety errors stop.
            if (exc.status in {401, 402, 403, 404} or exc.code == "content_filter"
                    or (not exc.retryable and exc.code not in {"input_limit", "output_limit"})):
                raise
            task = self.exp_config.task_name.lower()
            require_description(data.get(f"target_{task}_desc0"), "规划")
            data.pop(f"target_{task}_stylist_desc0", None)
            data["stylist_status"] = "fallback"
            data["stylist_error_code"] = exc.code
            if isinstance(exc.status, int):
                data["stylist_http_status"] = exc.status
            data["stylist_warning"] = f"风格优化未完成：{exc.safe_hint}。已使用原规划描述继续生成，本候选未应用风格优化。"
            self._progress(data, "stylist_fallback")
            return data

    async def _retrieve(self, data: dict) -> dict:
        setting = self.exp_config.retrieval_setting
        if setting in {"random", "none"}:
            return await self._stage(data, "retriever", self.retriever_agent, retrieval_setting=setting)
        path = reference_root(self.exp_config) / ("agent_selected_12.json" if setting == "manual" else "ref.json")
        stat = path.stat() if path.is_file() else None
        identity = json.dumps({"content": data.get("content"), "intent": data.get("visual_intent"),
                               "setting": setting, "library": str(path),
                               "version": (stat.st_mtime_ns, stat.st_size) if stat else None,
                               "model": self.exp_config.model_name}, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        lock = self._retrieval_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._retrieval_cache:
                data.update(deepcopy(self._retrieval_cache[key]))
                data["retrieval_cache_hit"] = True
                self._progress(data, "retriever_cached")
                return data
            data = await self._stage(data, "retriever", self.retriever_agent, retrieval_setting=setting)
            fields = ("top10_references", "retrieved_examples", "retrieval_status", "retrieval_count")
            self._retrieval_cache[key] = {field: deepcopy(data[field]) for field in fields if field in data}
            data["retrieval_cache_hit"] = False
            return data

    async def _run_critic_iterations(self, data: Dict[str, Any], task_name: str,
                                     max_rounds: int = 3, source: str = "stylist") -> Dict[str, Any]:
        initial = f"target_{task_name}_desc0" if source == "planner" else f"target_{task_name}_stylist_desc0"
        best = initial+"_base64_jpg"
        if data.get("status") == "code_only":
            return data
        if not data.get(best):
            raise ValueError("初始图片生成失败")
        data["eval_image_field"] = best
        for index in range(max_rounds):
            data["current_critic_round"] = index
            data = await self._stage(data, f"critic_{index+1}", self.critic_agent, source=source)
            suggestions = data.get(f"target_{task_name}_critic_suggestions{index}")
            if not isinstance(suggestions, str) or not suggestions.strip():
                raise ValueError("评审建议格式无效")
            if suggestions.strip() == "No changes needed.":
                break
            data = await self._stage(data, f"visualizer_{index+1}", self.visualizer_agent)
            output = f"target_{task_name}_critic_desc{index}_base64_jpg"
            if not data.get(output):
                raise ValueError("评审修改后的图片生成失败")
            best = output
            data["eval_image_field"] = best
        data["eval_image_field"] = best
        return data

    async def process_single_query(self, data: Dict[str, Any], do_eval=True) -> Dict[str, Any]:
        mode = self.exp_config.exp_mode
        task = self.exp_config.task_name.lower()
        data["status"] = "running"
        if mode == "vanilla":
            data = await self._stage(data, "visualizer", self.vanilla_agent)
            data["eval_image_field"] = f"vanilla_{task}_base64_jpg"
        elif mode == "dev_polish":
            data = await self._stage(data, "polish", self.polish_agent)
            data["eval_image_field"] = f"polished_{task}_base64_jpg"
        elif mode == "dev_retriever":
            data = await self._retrieve(data)
            do_eval = False
        elif mode in {"dev_planner", "dev_planner_stylist", "dev_planner_critic", "demo_planner_critic", "dev_full", "demo_full"}:
            data = await self._retrieve(data)
            data = await self._stage(data, "planner", self.planner_agent)
            stylist = mode in {"dev_planner_stylist", "dev_full", "demo_full"}
            if stylist:
                data = await self._style(data)
                stylist = data.get("stylist_status") != "fallback"
            data = await self._stage(data, "visualizer", self.visualizer_agent)
            data["eval_image_field"] = f"target_{task}_stylist_desc0_base64_jpg" if stylist else f"target_{task}_desc0_base64_jpg"
            if mode in {"dev_planner_critic", "demo_planner_critic", "dev_full", "demo_full"}:
                data = await self._run_critic_iterations(data, task, resolve_rounds(data, self.exp_config), "stylist" if stylist else "planner")
        else:
            raise ValueError("未知流水线模式")
        if data.get("status") == "code_only":
            self._progress(data, "code_ready")
            return data
        if mode != "dev_retriever" and not data.get(data.get("eval_image_field", "")):
            raise ValueError("候选没有可用图片")
        if do_eval and not mode.startswith("demo"):
            self._progress(data, "evaluation")
            data = await self.evaluation_function(data, exp_config=self.exp_config)
        data["status"] = "completed"
        self._progress(data, "completed")
        return data

    async def process_queries_batch(self, data_list: List[Dict[str, Any]], max_concurrent: int = 3,
                                    do_eval: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        async def process_one(original: dict) -> dict:
            data = deepcopy(original)
            async with semaphore:
                try:
                    return await self.process_single_query(data, do_eval=do_eval)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    stage = data.get("stage", "pipeline")
                    data["failed_stage"] = stage
                    data["status"] = "failed"
                    # Never persist exception text: providers can include keys, URLs or user text.
                    data["error"] = f"{stage} 阶段失败（{type(exc).__name__}）"
                    status = getattr(exc, "status", None)
                    if isinstance(exc, ProviderRequestError):
                        data["error_code"] = exc.code
                        data["error"] = f"{stage} 阶段失败：{exc.safe_hint}"
                        if isinstance(status, int):
                            data["http_status"] = status
                    elif isinstance(status, int):
                        hints = {401: "密钥无效或未授权", 403: "该密钥没有访问权限", 404: "模型或接口路径不存在",
                                 429: "服务商限流或额度不足"}
                        data["error"] = f"{stage} 阶段失败：{hints.get(status, '服务商请求失败')}（HTTP {status}）"
                    elif type(exc).__name__ == "ImageBudgetExceeded":
                        data["error"] = "已达到本次生图请求上限，最后成功图片已保留。"
                    elif type(exc).__name__ == "ImagePromptTooLong":
                        data["error"] = "生图描述压缩后仍过长，未提交生图；请缩短描述或拆分子图。"
                    elif getattr(exc, "stage", None) == "download":
                        data["error"] = "图片下载失败，生成任务不会自动重新提交；请从服务商任务记录恢复图片。"
                    self._progress(data, "failed")
                    return data

        tasks = [asyncio.create_task(process_one(data)) for data in data_list]
        try:
            for future in asyncio.as_completed(tasks):
                yield await future
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def evaluation_function(self, data: Dict[str, Any], exp_config) -> Dict[str, Any]:
        from .eval_toolkits import get_score_for_image_referenced
        return await get_score_for_image_referenced(data, task_name=exp_config.task_name, work_dir=exp_config.work_dir)
