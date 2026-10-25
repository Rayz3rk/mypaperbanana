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

"""Direct figure generation through independently configured model connections."""
import json
from typing import Any, Dict
from utils import generation_utils
from utils.output_options import output_constraints, render_plot, require_response, store_png
from .base_agent import BaseAgent


class VanillaAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_name = self.exp_config.task_name
        self.is_plot = self.task_name == "plot"
        self.model_name = self.exp_config.model_name if self.is_plot else self.exp_config.image_model_name
        self.process_executor = None
        self.system_prompt = PLOT_VANILLA_AGENT_SYSTEM_PROMPT if self.is_plot else DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Source: {json.dumps(data['content'],ensure_ascii=False)}\nVisual intent: {data['visual_intent']}\nDo not include figure titles.\n{output_constraints(self.exp_config)}"
        key = f"vanilla_{self.task_name}"
        if self.is_plot:
            responses = await generation_utils.call_text_async(
                model_name=self.model_name, contents=[{"type": "text", "text": prompt+"\nReturn only executable Python matplotlib code. Preserve every data point. Do not use files, networking or plt.show()."}],
                config={"system_prompt": self.system_prompt, "temperature": self.exp_config.temperature, "max_output_tokens": 16000},
                max_attempts=3, retry_delay=5, error_context="vanilla_plot",
            )
            code = require_response(responses, "统计图代码")
            data[key+"_code"] = code
            data["code_field"] = key+"_code"
            if not getattr(self.exp_config, "allow_plot_code_execution", False):
                data["status"] = "code_only"
                return data
            formats = await render_plot(code, self.exp_config)
            store_png(data, key, formats["png"])
            for extension in ("svg", "pdf"):
                data[key+"_base64_"+extension] = formats[extension]
        else:
            responses = await generation_utils.call_image_async(
                model_name=self.model_name, prompt=prompt,
                config={"aspect_ratio": data.get("additional_info", {}).get("rounded_ratio", "16:9"), "quality": getattr(self.exp_config, "image_quality", "2K")},
                max_attempts=1, retry_delay=5, error_context="vanilla",
            )
            store_png(data, key, require_response(responses, "图像"))
        return data


DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to generate a high-quality scientific diagram that effectively illustrates the method described in the text, as the caption requires, and adhering strictly to modern academic visualization standards.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## INPUT DATA
-   **Method Section**: [Content of method section]
-   **Diagram Caption**: [Diagram caption]
## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns well with the caption.
"""

PLOT_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are an expert statistical plot illustrator for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with "Plot Raw Data" and a "Visual Intent of the Desired Plot". Your task is to write matplotlib code to generate a high-quality statistical plot that effectively visualizes the data according to the visual intent, adhering strictly to modern academic visualization standards.

## INPUT DATA
-   **Plot Raw Data**: [Raw data to be visualized]
-   **Visual Intent of the Desired Plot**: [Description of what the plot should convey]

## OUTPUT
Write Python matplotlib code to generate the plot. Only provide the code without any explanations.
"""
