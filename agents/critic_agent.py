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

"""Critique rendered figures with strictly validated structured responses."""
import json
from typing import Any, Dict
import json_repair
from utils import generation_utils
from utils.output_options import image_media_type, output_constraints, require_description, require_response
from .base_agent import BaseAgent


class CriticAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name
        self.task_name = self.exp_config.task_name
        self.system_prompt = PLOT_CRITIC_AGENT_SYSTEM_PROMPT if self.task_name == "plot" else DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT

    async def process(self, data: Dict[str, Any], source: str = "stylist") -> Dict[str, Any]:
        task = self.task_name
        index = data.get("current_critic_round", 0)
        if index:
            key = f"target_{task}_critic_desc{index-1}"
        elif source == "planner":
            key = f"target_{task}_desc0"
        elif source == "stylist":
            key = f"target_{task}_stylist_desc0"
        else:
            raise ValueError("未知评审来源")
        description = require_description(data.get(key), "待评审描述")
        image = data.get(key+"_base64_png") or data.get(key+"_base64_jpg")
        if not image:
            raise ValueError("缺少可评审图片")
        prompt = f"Detailed Description: {description}\nSource: {json.dumps(data['content'], ensure_ascii=False)}\nVisual Intent: {data['visual_intent']}\n{output_constraints(self.exp_config)}\nReturn JSON with string fields critic_suggestions and revised_description. Use exactly 'No changes needed.' for both only if the figure passes all checks."
        if self.task_name == "diagram":
            prompt += ("\nWhen revising, keep revised_description complete but concise: about 800 "
                       "Chinese characters or 450 English words at most. Retain correct labels, "
                       "dimensions, connections and scientific constraints; avoid repeating the source.")
        contents = [{"type": "text", "text": prompt}, {"type": "image", "source": {"type": "base64", "data": image, "media_type": image_media_type(image)}}]
        for attempt in range(2):
            responses = await generation_utils.call_text_async(
                model_name=self.model_name, contents=contents,
                config={"system_prompt": self.system_prompt, "temperature": self.exp_config.temperature, "max_output_tokens": 16000},
                max_attempts=3, retry_delay=5, error_context="critic",
            )
            raw = require_response(responses, "评审")
            parsed = json_repair.loads(raw.replace("```json", "").replace("```", "").strip())
            if isinstance(parsed, dict) and all(isinstance(parsed.get(field), str) and parsed[field].strip() for field in ("critic_suggestions", "revised_description")):
                suggestions = parsed["critic_suggestions"].strip()
                revised = parsed["revised_description"].strip()
                if (revised == "No changes needed.") != (suggestions == "No changes needed."):
                    if attempt == 0:
                        contents.append({"type": "text", "text": "The previous response was inconsistent. Return a revised description implementing your suggestions, or mark both fields No changes needed."})
                        continue
                    raise ValueError("评审响应字段不一致")
                data[f"target_{task}_critic_suggestions{index}"] = suggestions
                data[f"target_{task}_critic_desc{index}"] = description if revised == "No changes needed." else revised
                return data
            contents.append({"type": "text", "text": "The previous response had an invalid schema. Return a JSON object containing the two required nonempty string fields."})
        raise ValueError("评审返回格式无效，不能视为审核通过")


DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Your task is to conduct a sanity check and provide a critique of the target diagram based on its content and presentation. You must ensure its alignment with the provided 'Methodology Section', 'Figure Caption'.

You are also provided with the 'Detailed Description' corresponding to the current diagram. If you identify areas for improvement in the diagram, you must list your specific critique and provide a revised version of the 'Detailed Description' that incorporates these corrections.

## CRITIQUE & REVISION RULES

1. Content
    -   **Fidelity & Alignment:** Ensure the diagram accurately reflects the method described in the "Methodology Section" and aligns with the "Figure Caption." Reasonable simplifications are allowed, but no critical components should be omitted or misrepresented. Also, the diagram should not contain any hallucinated content. Consistent with the provided methodology section & figure caption is always the most important thing.
    -   **Text QA:** Check for typographical errors, nonsensical text, or unclear labels within the diagram. Suggest specific corrections.
    -   **Validation of Examples:** Verify the accuracy of illustrative examples. If the diagram includes specific examples to aid understanding (e.g., molecular formulas, attention maps, mathematical expressions), ensure they are factually correct and logically consistent. If an example is incorrect, provide the correct version.
    -   **Caption Exclusion:** Ensure the figure caption text (e.g., "Figure 1: Overview...") is **not** included within the image visual itself. The caption should remain separate.

2. Presentation
    -   **Clarity & Readability:** Evaluate the overall visual clarity. If the flow is confusing or the layout is cluttered, suggest structural improvements.
    -   **Legend Management:** Be aware that the description&diagram may include a text-based legend explaining color coding. Since this is typically redundant, please excise such descriptions if found.

** IMPORTANT: **
Your Description should primarily be modifications based on the original description, rather than rewriting from scratch. If the original description has obvious problems in certain parts that require re-description, your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background, colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.

## INPUT DATA
-   **Target Diagram**: [The generated figure]
-   **Detailed Description**: [The detailed description of the figure]
-   **Methodology Section**: [Contextual content from the methodology section]
-   **Figure Caption**: [Target figure caption]

## OUTPUT
Provide your response strictly in the following JSON format.

```json
{
    "critic_suggestions": "Insert your detailed critique and specific suggestions for improvement here. If the diagram is perfect, write 'No changes needed.'",
    "revised_description": "Insert the fully revised detailed description here, incorporating all your suggestions. If no changes are needed, write 'No changes needed.'",
}
```
"""

PLOT_CRITIC_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Your task is to conduct a sanity check and provide a critique of the target plot based on its content and presentation. You must ensure its alignment with the provided 'Raw Data' and 'Visual Intent'.

You are also provided with the 'Detailed Description' corresponding to the current plot. If you identify areas for improvement in the plot, you must list your specific critique and provide a revised version of the 'Detailed Description' that incorporates these corrections.

## CRITIQUE & REVISION RULES

1. Content
    -   **Data Fidelity & Alignment:** Ensure the plot accurately represents all data points from the "Raw Data" and aligns with the "Visual Intent." All quantitative values must be correct. No data should be hallucinated, omitted, or misrepresented.
    -   **Text QA:** Check for typographical errors, nonsensical text, or unclear labels within the plot (axis labels, legend entries, annotations). Suggest specific corrections.
    -   **Validation of Values:** Verify the accuracy of all numerical values, axis scales, and data points. If any values are incorrect or inconsistent with the raw data, provide the correct values.
    -   **Caption Exclusion:** Ensure the figure caption text (e.g., "Figure 1: Performance comparison...") is **not** included within the image visual itself. The caption should remain separate.

2. Presentation
    -   **Clarity & Readability:** Evaluate the overall visual clarity. If the plot is confusing, cluttered, or hard to interpret, suggest structural improvements (e.g., better axis labeling, clearer legend, appropriate plot type).
    -   **Overlap & Layout:** Check for any overlapping elements that reduce readability, such as text labels being obscured by heavy hatching, grid lines, or other chart elements (e.g., pie chart labels inside dark slices). If overlaps exist, suggest adjusting element positions (e.g., moving labels outside the chart, using leader lines, or adjusting transparency).
    -   **Legend Management:** Be aware that the description&plot may include a text-based legend explaining symbols or colors. Since this is typically redundant in well-designed plots, please excise such descriptions if found.

3. Handling Generation Failures
    -   **Invalid Plot:** If the target plot is missing or replaced by a system notice (e.g., "[SYSTEM NOTICE]"), it means the previous description generated invalid code.
    -   **Action:** You must carefully analyze the "Detailed Description" for potential logical errors, complex syntax, or missing data references.
    -   **Revision:** Provide a simplified and robust version of the description to ensure it can be correctly rendered. Do not just repeat the same description.

## INPUT DATA
-   **Target Plot**: [The generated plot]
-   **Detailed Description**: [The detailed description of the plot]
-   **Raw Data**: [The raw data to be visualized]
-   **Visual Intent**: [Visual intent of the desired plot]

## OUTPUT
Provide your response strictly in the following JSON format.

```json
{
    "critic_suggestions": "Insert your detailed critique and specific suggestions for improvement here. If the plot is perfect, write 'No changes needed.'",
    "revised_description": "Insert the fully revised detailed description here, incorporating all your suggestions. If no changes are needed, write 'No changes needed.'",
}
```
"""
