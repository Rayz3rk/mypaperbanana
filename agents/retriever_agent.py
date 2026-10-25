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

"""Retrieve valid reference figures; empty libraries never call a model."""
import json
import random
from typing import Any, Dict
import json_repair
from utils import generation_utils
from utils.output_options import reference_root, require_response
from .base_agent import BaseAgent


class RetrieverAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name
        self.task_name = self.exp_config.task_name
        self.system_prompt = PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT if self.task_name == "plot" else DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT

    def _pool(self, manual: bool = False) -> list[dict]:
        root = reference_root(self.exp_config)
        path = root / ("agent_selected_12.json" if manual else "ref.json")
        if not path.is_file():
            return []
        try:
            pool = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return []
        if not isinstance(pool, list):
            return []
        valid = []
        for item in pool:
            if not isinstance(item, dict) or not all(key in item for key in ("id", "visual_intent", "path_to_gt_image")):
                continue
            image = (root / str(item["path_to_gt_image"])).resolve()
            if image.is_relative_to(root.resolve()) and image.is_file():
                valid.append(item)
        return valid

    async def process(self, data: Dict[str, Any], retrieval_setting: str = "auto") -> Dict[str, Any]:
        if retrieval_setting not in {"auto", "auto-full", "manual", "random", "none"}:
            raise ValueError("未知检索模式")
        data["top10_references"] = []
        data["retrieved_examples"] = []
        data["retrieval_status"] = "disabled"
        data["retrieval_count"] = 0
        if retrieval_setting == "none":
            return data
        pool = self._pool(manual=retrieval_setting == "manual")
        if not pool:
            data["retrieval_status"] = "empty_or_missing"
            return data
        if retrieval_setting == "manual":
            selected = pool[:10]
        elif retrieval_setting == "random":
            selected = random.sample(pool, min(10, len(pool)))
        else:
            pool = pool[:200] if self.task_name == "diagram" else pool
            prompt_pool = [{"id": item["id"], "caption": item["visual_intent"], **({"content": item.get("content", "")} if retrieval_setting == "auto-full" else {})} for item in pool]
            prompt = f"Select at most 10 relevant reference IDs for this source: {data['content']}\nFigure intent: {data['visual_intent']}\nCandidates: {json.dumps(prompt_pool,ensure_ascii=False)}\nReturn JSON with a list field top10_references, containing only candidate IDs."
            responses = await generation_utils.call_text_async(
                model_name=self.model_name, contents=[{"type": "text", "text": prompt}],
                config={"system_prompt": self.system_prompt, "temperature": self.exp_config.temperature, "max_output_tokens": 2000},
                max_attempts=3, retry_delay=5, error_context="retriever",
            )
            ids = self._parse_retrieval_result(require_response(responses, "参考检索"), self.task_name)
            selected = [item for item in pool if item["id"] in ids][:10]
        data["top10_references"] = [item["id"] for item in selected]
        data["retrieved_examples"] = selected
        data["retrieval_status"] = "completed"
        data["retrieval_count"] = len(selected)
        return data

    def _parse_retrieval_result(self, raw_response: str, task_name: str) -> list:
        parsed = json_repair.loads(raw_response)
        if isinstance(parsed, list):
            return parsed
        if not isinstance(parsed, dict):
            raise ValueError("参考检索返回格式无效")
        for key in ("top10_references", "top10_diagrams", "top10_plots"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        raise ValueError("参考检索未返回有效 ID 列表")


DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate method diagrams for academic papers**. Given a paper's methodology section and a figure caption, the system needs to create a high-quality illustrative diagram that visualizes the described method.

To help the AI learn how to generate appropriate diagrams, we use a **few-shot learning approach**: we provide it with reference examples of similar diagrams. The AI will learn from these examples to understand what kind of diagram to create for the target.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference diagrams from a candidate pool that will serve as few-shot examples for the diagram generation model.

You will receive:
- **Target Input:** The methodology section and caption of the diagram we need to generate
- **Candidate Pool:** ~200 existing diagrams (each with methodology and caption)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to draw the target diagram.

# Selection Logic (Topic + Intent)

Your goal is to find examples that match the Target in both **Domain** and **Diagram Type**.

**1. Match Research Topic (Use Methodology & Caption):**
* What is the domain? (e.g., Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications).
* Select candidates that belong to the **same research domain**.
* *Why?* Similar domains share similar terminology (e.g., "Actor-Critic" in RL).

**2. Match Visual Intent (Use Caption & Keywords):**
* What type of diagram is implied? (e.g., "Framework", "Pipeline", "Detailed Module", "Performance Chart").
* Select candidates with **similar visual structures**.
* *Why?* A "Framework" diagram example is useless for drawing a "Performance Bar Chart", even if they are in the same domain.

**Ranking Priority:**
1.  **Best Match:** Same Topic AND Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Agent Framework", Target is "Dataset Construction Pipeline" -> Candidate is "Dataset Construction Pipeline").
2.  **Second Best:** Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Vision Framework"). *Structure is more important than Topic for drawing.*
3.  **Avoid:** Different Visual Intent (e.g., Target is "Pipeline" -> Candidate is "Bar Chart").

# Input Data

## Target Input
-   **Caption:** [Caption of the target diagram]
-   **Methodology section:** [Methodology section of the target paper]

## Candidate Pool
List of candidate diagrams, each structured as follows:

Candidate Diagram i:
-   **Diagram ID:** [ID of the candidate diagram (ref_1, ref_2, ...)]
-   **Caption:** [Caption of the candidate diagram]
-   **Methodology section:** [Methodology section of the candidate's paper]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact IDs** of the Top 10 selected diagrams (use the exact IDs from the Candidate Pool, such as "ref_1", "ref_25", "ref_100", etc.):
```json
{
  "top10_diagrams": [
    "ref_1",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""

PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate statistical plots**. Given a plot's raw data and the visual intent, the system needs to create a high-quality visualization that effectively presents the data.

To help the AI learn how to generate appropriate plots, we use a **few-shot learning approach**: we provide it with reference examples of similar plots. The AI will learn from these examples to understand what kind of plot to create for the target data.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference plots from a candidate pool that will serve as few-shot examples for the plot generation model.

You will receive:
- **Target Input:** The raw data and visual intent of the plot we need to generate
- **Candidate Pool:** Reference plots (each with raw data and visual intent)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to create the target plot.

# Selection Logic (Data Type + Visual Intent)

Your goal is to find examples that match the Target in both **Data Characteristics** and **Plot Type**.

**1. Match Data Characteristics (Use Raw Data & Visual Intent):**
* What type of data is it? (e.g., categorical vs numerical, single series vs multi-series, temporal vs comparative).
* What are the data dimensions? (e.g., 1D, 2D, 3D).
* Select candidates with **similar data structures and characteristics**.
* *Why?* Different data types require different visualization approaches.

**2. Match Visual Intent (Use Visual Intent):**
* What type of plot is implied? (e.g., "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart").
* Select candidates with **similar plot types**.
* *Why?* A "bar chart" example is more useful for generating another bar chart than a "scatter plot" example, even if the data domains are similar.

**Ranking Priority:**
1.  **Best Match:** Same Data Type AND Same Plot Type (e.g., Target is "multi-series line chart" -> Candidate is "multi-series line chart").
2.  **Second Best:** Same Plot Type with compatible data (e.g., Target is "bar chart with 5 categories" -> Candidate is "bar chart with 6 categories").
3.  **Avoid:** Different Plot Type (e.g., Target is "bar chart" -> Candidate is "pie chart"), unless there are no more candidates with the same plot type.

# Input Data

## Target Input
-   **Visual Intent:** [Visual intent of the target plot]
-   **Raw Data:** [Raw data to be visualized]

## Candidate Pool
List of candidate plots, each structured as follows:

Candidate Plot i:
-   **Plot ID:** [ID of the candidate plot (ref_0, ref_1, ...)]
-   **Visual Intent:** [Visual intent of the candidate plot]
-   **Raw Data:** [Raw data of the candidate plot]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Plot IDs** of the Top 10 selected plots (use the exact IDs from the Candidate Pool, such as "ref_0", "ref_25", "ref_100", etc.):
```json
{
  "top10_plots": [
    "ref_0",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""
