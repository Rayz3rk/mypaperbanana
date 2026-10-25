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

"""Apply style suggestions using separate text and image connections."""
import base64
from typing import Any, Dict
from utils import generation_utils
from utils.output_options import image_media_type, output_constraints, reference_root, require_response, store_png
from .base_agent import BaseAgent


class PolishAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_model_name = self.exp_config.image_model_name
        self.text_model_name = self.exp_config.model_name
        self.task_name = self.exp_config.task_name
        self.suggestion_system_prompt = PLOT_SUGGESTION_SYSTEM_PROMPT if self.task_name == "plot" else DIAGRAM_SUGGESTION_SYSTEM_PROMPT

    async def _generate_suggestions(self, gt_image_b64: str, style_guide: str) -> str:
        responses = await generation_utils.call_text_async(
            model_name=self.text_model_name,
            contents=[{"type": "text", "text": f"Style guide: {style_guide}\nList specific improvements or say No changes needed.\n{output_constraints(self.exp_config)}"}, {"type": "image", "source": {"type": "base64", "data": gt_image_b64, "media_type": image_media_type(gt_image_b64)}}],
            config={"system_prompt": self.suggestion_system_prompt, "temperature": self.exp_config.temperature, "max_output_tokens": 4000},
            max_attempts=3, retry_delay=5, error_context="polish_suggestions",
        )
        return require_response(responses, "精修建议")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        root = reference_root(self.exp_config)
        relative = data.get("path_to_gt_image")
        if not relative:
            raise ValueError("缺少待精修原图")
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError("待精修图片不存在")
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        guide = self.exp_config.work_dir / f"style_guides/neurips2025_{self.task_name}_style_guide.md"
        style = guide.read_text(encoding="utf-8") if guide.is_file() else "Improve readability while preserving every scientific detail."
        suggestions = await self._generate_suggestions(encoded, style)
        data[f"suggestions_{self.task_name}"] = suggestions
        key = f"polished_{self.task_name}"
        if suggestions.strip().rstrip(".") == "No changes needed":
            store_png(data, key, encoded)
            return data
        media = image_media_type(encoded)
        responses = await generation_utils.call_image_async(
            model_name=self.image_model_name,
            prompt=f"Polish the provided reference image according to these suggestions: {suggestions}\nPreserve all scientific content.\n{output_constraints(self.exp_config)}",
            config={"aspect_ratio": data.get("additional_info", {}).get("rounded_ratio", "16:9"), "quality": getattr(self.exp_config, "image_quality", "2K"), "image_urls": [f"data:{media};base64,{encoded}"]},
            max_attempts=1, retry_delay=5, error_context="polish_image",
        )
        store_png(data, key, require_response(responses, "图像精修"))
        return data


DIAGRAM_SUGGESTION_SYSTEM_PROMPT = """
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed".
"""

PLOT_SUGGESTION_SYSTEM_PROMPT = """
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed".
"""

DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions.
"""

PLOT_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement suggestions. Your task is to generate a polished version of this plot by applying these suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information in the plot.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions (colors, fonts, layout, etc.).
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards for top-tier conferences.

## OUTPUT
Generate a polished plot image that maintains the original data while applying the improvement suggestions.
"""
