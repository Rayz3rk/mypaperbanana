"""Consistent result selection, lossless downloads and atomic local history."""

import base64
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "demo"


def image_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("图片数据必须是 base64 字符串。")
    data = base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value, validate=True)
    with Image.open(BytesIO(data)) as image:
        image.verify()
    return data


def png_bytes(value: str) -> bytes:
    data = image_bytes(value)
    with Image.open(BytesIO(data)) as image:
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def final_image_key(result: dict, exp_mode: str = "") -> str | None:
    selected = result.get("eval_image_field")
    if isinstance(selected, str) and isinstance(result.get(selected), str) and result.get(selected):
        png = selected.removesuffix("_base64_jpg") + "_base64_png"
        return png if result.get(png) else selected
    critic = []
    for key, value in result.items():
        match = re.fullmatch(r"target_(?:diagram|plot)_critic_desc(\d+)_base64_(png|jpg)", key)
        if match and isinstance(value, str) and value:
            critic.append((int(match[1]), match[2] == "png", key))
    if critic:
        return max(critic)[2]
    for kind in ("stylist_desc0", "desc0", "vanilla", "polished"):
        for task in ("diagram", "plot"):
            for suffix in ("png", "jpg"):
                key = f"target_{task}_{kind}_base64_{suffix}"
                if isinstance(result.get(key), str) and result.get(key):
                    return key
    for prefix in ("vanilla_diagram", "vanilla_plot", "polished_diagram", "polished_plot"):
        for suffix in ("png", "jpg"):
            key = f"{prefix}_base64_{suffix}"
            if isinstance(result.get(key), str) and result.get(key):
                return key
    return None


def evolution_stages(result: dict, exp_mode: str = "") -> list[dict]:
    stages = []
    seen = set()
    keys = []
    for key, value in result.items():
        match = re.fullmatch(r"(target_(?:diagram|plot)_(desc0|stylist_desc0|critic_desc(\d+)))_base64_(png|jpg)", key)
        if match and isinstance(value, str) and value:
            order = int(match[3]) + 2 if match[3] is not None else (1 if match[2] == "stylist_desc0" else 0)
            keys.append((order, match[4] != "png", key, match))
    for order, _, key, match in sorted(keys, key=lambda item: item[:3]):
        prefix = match[1]
        if prefix in seen:
            continue
        seen.add(prefix)
        name = "规划结果" if order == 0 else "风格优化" if order == 1 else f"评审第 {order - 1} 轮"
        stages.append({"name": name, "image_key": key, "desc_key": prefix,
                       "suggestions_key": prefix.replace("critic_desc", "critic_suggestions"),
                       "description": name})
    return stages


def result_downloads(result: dict, exp_mode: str = "") -> dict[str, bytes]:
    key = final_image_key(result, exp_mode)
    if not key:
        return {}
    downloads = {"png": png_bytes(result[key])}
    prefix = re.sub(r"_base64_(png|jpg)$", "", key)
    for extension in ("svg", "pdf"):
        encoded = result.get(f"{prefix}_base64_{extension}")
        if encoded:
            downloads[extension] = base64.b64decode(encoded, validate=True)
    return downloads


def results_zip(results: list[dict], exp_mode: str = "") -> bytes:
    buffer = BytesIO()
    used = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, result in enumerate(results):
            identity = result.get("candidate_id", index)
            number = identity + 1 if isinstance(identity, int) and not isinstance(identity, bool) and identity >= 0 else index + 1
            name = f"candidate_{number}"
            if name in used:
                name += f"_result_{index + 1}"
            used.add(name)
            for extension, data in result_downloads(result, exp_mode).items():
                archive.writestr(f"{name}.{extension}", data)
    return buffer.getvalue()


def normalize_history(payload) -> dict:
    if isinstance(payload, list):
        payload = {"version": 1, "results": payload, "settings": {}, "status": "completed"}
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("结果文件必须包含 results 列表，或使用旧版结果数组。")
    if not all(isinstance(result, dict) for result in payload["results"]):
        raise ValueError("候选结果必须是对象。")
    payload = dict(payload)
    for name in ("settings", "usage", "stages"):
        value = payload.setdefault(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"结果文件的 {name} 字段必须是对象。")
    inputs = payload.setdefault("inputs", [])
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise ValueError("结果文件的 inputs 字段必须是对象列表。")
    for name in ("status", "created_at", "error"):
        if name in payload and not isinstance(payload[name], str):
            raise ValueError(f"结果文件的 {name} 字段必须是字符串。")
    return payload


def write_history(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_history(path: Path, root: Path = RESULTS_DIR) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved.suffix != ".json":
        raise ValueError("只能载入本地结果目录中的 JSON 文件。")
    return normalize_history(json.loads(resolved.read_text(encoding="utf-8")))


def list_history(root: Path = RESULTS_DIR) -> list[Path]:
    if not root.exists():
        return []
    return sorted((path for path in root.glob("*.json") if path.is_file() and
                   path.resolve().is_relative_to(root.resolve())), reverse=True)
