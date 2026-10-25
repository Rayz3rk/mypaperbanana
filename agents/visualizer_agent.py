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

"""Render each requested stage once, retaining lossless publication formats."""
from typing import Any, Dict
from utils import generation_utils
from utils.output_options import output_constraints, render_plot, require_description, require_response, store_png
from .base_agent import BaseAgent


class VisualizerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_name = self.exp_config.task_name
        self.is_plot = self.task_name == "plot"
        self.model_name = self.exp_config.model_name if self.is_plot else self.exp_config.image_model_name
        self.system_prompt = PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT if self.is_plot else DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT
        self.process_executor = None

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        task = self.task_name
        if "current_critic_round" in data:
            keys = [f"target_{task}_critic_desc{data['current_critic_round']}"]
        elif f"target_{task}_stylist_desc0" in data:
            keys = [f"target_{task}_stylist_desc0"]
            if getattr(self.exp_config, "keep_intermediate_images", False):
                keys.insert(0, f"target_{task}_desc0")
        else:
            keys = [f"target_{task}_desc0"]
        for key in keys:
            if data.get(key+"_base64_png") or data.get(key+"_base64_jpg"):
                continue
            description = require_description(data.get(key), "图像描述")
            prompt = f"Render a clear scientific figure from this description: {description}\nDo not include figure titles.\n{output_constraints(self.exp_config)}"
            if self.is_plot:
                responses = await generation_utils.call_text_async(
                    model_name=self.model_name,
                    contents=[{"type": "text", "text": prompt+"\nReturn only executable Python matplotlib code, including every original data point. Do not read or write external files, access the network, or call plt.show()."}],
                    config={"system_prompt": self.system_prompt, "temperature": self.exp_config.temperature, "max_output_tokens": 16000},
                    max_attempts=3, retry_delay=5, error_context="plot_code",
                )
                code = require_response(responses, "统计图代码")
                data[key+"_code"] = code
                data["code_field"] = key+"_code"
                if not getattr(self.exp_config, "allow_plot_code_execution", False):
                    data["status"] = "code_only"
                    continue
                formats = await render_plot(code, self.exp_config)
                store_png(data, key, formats["png"])
                for extension in ("svg", "pdf"):
                    data[key+"_base64_"+extension] = formats[extension]
            else:
                responses = await generation_utils.call_image_async(
                    model_name=self.model_name, prompt=prompt,
                    config={"aspect_ratio": data.get("additional_info", {}).get("rounded_ratio", "16:9"), "quality": getattr(self.exp_config, "image_quality", "2K")},
                    max_attempts=1, retry_delay=5, error_context="visualizer",
                )
                store_png(data, key, require_response(responses, "图像"))
        return data


DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests."""

PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert statistical plot illustrator. Write code to generate high-quality statistical plots based on user requests."""
