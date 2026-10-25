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

"""Plan figures using provider-independent multimodal inputs."""
import base64
import json
from typing import Any, Dict
from utils import generation_utils
from utils.output_options import image_media_type, output_constraints, reference_root, require_response
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name
        plot = self.exp_config.task_name == "plot"
        self.system_prompt = PLOT_PLANNER_AGENT_SYSTEM_PROMPT if plot else DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT
        self.task_config = {"task_name": "plot" if plot else "diagram", "content_label": "Plot Raw Data" if plot else "Methodology Section", "visual_intent_label": "Visual Intent" if plot else "Diagram Caption"}

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        root = reference_root(self.exp_config)
        examples = data.get("retrieved_examples", [])
        if not examples and data.get("top10_references") and (root / "ref.json").is_file():
            pool = json.loads((root / "ref.json").read_text(encoding="utf-8"))
            lookup = {item["id"]: item for item in pool if isinstance(item, dict) and "id" in item}
            examples = [lookup[item] for item in data["top10_references"] if item in lookup]
        contents = []
        for index, item in enumerate(examples):
            image_path = (root / item["path_to_gt_image"]).resolve()
            if not image_path.is_relative_to(root.resolve()) or not image_path.is_file():
                continue
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            contents.extend([
                {"type": "text", "text": f"Example {index+1}:\n{cfg['content_label']}: {item.get('content', '')}\n{cfg['visual_intent_label']}: {item.get('visual_intent', '')}"},
                {"type": "image", "source": {"type": "base64", "data": encoded, "media_type": image_media_type(encoded)}},
            ])
        raw = data["content"]
        content = json.dumps(raw, ensure_ascii=False) if isinstance(raw, (dict, list)) else str(raw)
        prompt = f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {data['visual_intent']}\nProvide a detailed description of the target figure. Do not include figure titles.\n{output_constraints(self.exp_config)}"
        if self.exp_config.task_name == "diagram":
            prompt += ("\nFor the image API, keep the COMPLETE description concise: aim for 800 Chinese "
                       "characters (about 450 English words) or fewer. Keep labels, exact dimensions, "
                       "connections and scientific constraints. No repeated style rules or explanations.")
        contents.append({"type": "text", "text": prompt})
        responses = await generation_utils.call_text_async(
            model_name=self.model_name, contents=contents,
            config={"system_prompt": self.system_prompt, "temperature": self.exp_config.temperature, "max_output_tokens": 16000},
            max_attempts=3, retry_delay=5, error_context="planner",
        )
        data[f"target_{cfg['task_name']}_desc0"] = require_response(responses, "规划")
        return data


DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.
"""

PLOT_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"""
