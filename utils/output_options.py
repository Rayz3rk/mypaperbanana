"""Shared output validation, language constraints, and publication formats."""

import asyncio
import base64
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from PIL import Image


def require_response(responses: list[str], stage: str) -> str:
    """Reject legacy error sentinels before another paid stage starts."""
    if not responses or not isinstance(responses[0], str):
        raise ValueError(f"{stage} 未返回有效内容")
    response = responses[0].strip()
    if not response or response.casefold() in {"error", "none", "null"}:
        raise ValueError(f"{stage} 生成失败")
    return response


def require_description(value: object, stage: str) -> str:
    return require_response([value] if isinstance(value, str) else [], stage)


def output_constraints(config: object) -> str:
    language = getattr(config, "output_language", "auto")
    labels = {
        "auto": "Match the figure label language to the user's caption and preserve its terminology.",
        "zh": "All figure labels must be Simplified Chinese; preserve standard symbols and necessary acronyms.",
        "en": "All figure labels must be English; preserve standard mathematical symbols.",
        "bilingual": "Use concise Chinese / English bilingual labels for all named components.",
    }
    text = labels.get(language, labels["auto"])
    text += " Check every visible label for spelling, legibility, missing glyphs and consistency. Use a CJK-capable font when Chinese is required."
    glossary = getattr(config, "glossary", "")
    if glossary:
        text += "\nUser terminology (use the exact supplied mappings):\n" + str(glossary)
    return text


def reference_root(config: object) -> Path:
    dataset = getattr(config, "reference_dataset_name", "") or getattr(config, "dataset_name", "PaperBananaBench")
    if dataset in {"Demo", "demo"}:
        dataset = "PaperBananaBench"
    return Path(config.work_dir) / "data" / dataset / config.task_name


def image_media_type(encoded: str) -> str:
    try:
        raw = base64.b64decode(encoded.split(",", 1)[-1])
        with Image.open(io.BytesIO(raw)) as image:
            return Image.MIME.get(image.format, "image/png")
    except (ValueError, OSError):
        raise ValueError("模型未返回有效图片") from None


def store_png(data: dict, description_key: str, encoded: str) -> None:
    """Keep a lossless PNG plus the legacy key used by existing history files."""
    raw = base64.b64decode(encoded.split(",", 1)[-1])
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
            image = image.convert("RGB")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
    png = base64.b64encode(stream.getvalue()).decode("ascii")
    data[f"{description_key}_base64_png"] = png
    data[f"{description_key}_base64_jpg"] = png
    data.setdefault("image_formats", {})[description_key] = "image/png"


def resolve_rounds(data: dict, config: object) -> int:
    value = data.get("max_critic_rounds")
    if value is None:
        value = getattr(config, "max_critic_rounds", 3)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
        raise ValueError("评审轮次必须为 0–10 的整数")
    return value


def _render_plot(code: str, timeout: float) -> dict[str, str]:
    match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
    source = match.group(1) if match else code
    # A disposable process and timeout limit lifetime, not filesystem permissions.
    # This path is explicitly opt-in because generated Python is executable code.
    worker = '''
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fonts = {item.name for item in font_manager.fontManager.ttflist}
cjk = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei") if name in fonts]
if cjk:
    plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
exec(Path("source.py").read_text(encoding="utf-8"), {"__name__": "__main__"})
if not plt.get_fignums():
    raise ValueError("Plot code did not create a figure")
for extension in ("png", "svg", "pdf"):
    plt.savefig("figure." + extension, format=extension, bbox_inches="tight", dpi=300)
plt.close("all")
'''
    with tempfile.TemporaryDirectory(prefix="paperbanana-plot-") as directory:
        path = Path(directory)
        (path / "source.py").write_text(source, encoding="utf-8")
        (path / "render.py").write_text(worker, encoding="utf-8")
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
        environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path / "render.py")],
                cwd=path, env=environment, timeout=timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("统计图代码执行超时") from None
        if completed.returncode or not (path / "figure.png").is_file():
            raise ValueError("统计图代码执行失败")
        return {
            extension: base64.b64encode((path / f"figure.{extension}").read_bytes()).decode("ascii")
            for extension in ("png", "svg", "pdf")
        }


async def render_plot(code: str, config: object) -> dict[str, str]:
    if not getattr(config, "allow_plot_code_execution", False):
        raise ValueError("统计图代码执行未启用；请显式启用后在可信本地环境运行")
    timeout = float(getattr(config, "plot_timeout_seconds", 30))
    return await asyncio.to_thread(_render_plot, code, max(1, min(timeout, 120)))
