"""Task-scoped model routing, cancellation and request budgets."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

from providers.connections import ModelConnection


class GenerationError(RuntimeError):
    """An upstream stage failed; downstream paid stages must stop."""


class ImageBudgetExceeded(GenerationError):
    pass


class ImagePromptTooLong(GenerationError):
    pass


# The observed Evolink model rejects prompts above 4000 tokens. Its exact
# tokenizer is not available locally; use a conservative UTF-8 byte budget,
# including all language/glossary boilerplate, rather than calling characters
# tokens. Compression is semantic, never a substring truncation.
EVOLINK_NANO_PROMPT_BYTES = 3600


@dataclass
class BoundSession:
    runtime: Any
    max_image_calls: int = 20
    max_output_tokens: int = 8192
    image_requests: int = 0
    text_requests: int = 0
    cancel_event: Any = None
    on_usage: Callable | None = None

    def check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise asyncio.CancelledError

    def usage(self) -> dict:
        values = dict(self.runtime.usage)
        values.update(image_requests=self.image_requests, text_requests=self.text_requests,
                      max_image_calls=self.max_image_calls)
        return values

    def notify(self) -> None:
        if self.on_usage:
            self.on_usage(self.usage())


_current_session: ContextVar[BoundSession | None] = ContextVar("paperbanana_model_session", default=None)


def has_model_session() -> bool:
    return _current_session.get() is not None


@asynccontextmanager
async def model_session(text: ModelConnection, image: ModelConnection, *, max_image_calls: int = 20,
                        max_output_tokens: int = 8192, cancel_event=None, on_usage=None):
    from providers.runtime import ModelRuntime

    async with ModelRuntime(text, image) as runtime:
        session = BoundSession(runtime, max_image_calls, max_output_tokens,
                               cancel_event=cancel_event, on_usage=on_usage)
        token = _current_session.set(session)
        try:
            yield session
        finally:
            session.notify()
            _current_session.reset(token)


def _config_dict(config) -> dict:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "model_dump"):
        return config.model_dump(exclude_none=True)
    return {key: getattr(config, key) for key in
            ("system_instruction", "temperature", "max_output_tokens", "image_config")
            if hasattr(config, key)}


def _valid_result(values: list[str], kind: str) -> list[str]:
    if not values or any(not isinstance(value, str) or not value.strip() or value.strip() == "Error"
                         for value in values):
        raise GenerationError(f"{kind}模型未返回有效结果，已停止此候选的后续请求。")
    return values


async def call_text_async(model_name, contents, config, max_attempts=3, retry_delay=2, error_context=""):
    session = _current_session.get()
    if session is None:
        from utils.settings import load_connections
        async with model_session(*load_connections()):
            return await call_text_async(model_name, contents, config, max_attempts, retry_delay, error_context)
    session.check_cancelled()
    cfg = _config_dict(config)
    session.text_requests += 1
    session.notify()
    try:
        values = await session.runtime.generate_text(
            model_name=session.runtime.text_connection.model or model_name, contents=contents,
            system_prompt=cfg.get("system_prompt", cfg.get("system_instruction", "")) or "",
            temperature=cfg.get("temperature", 1.0),
            max_output_tokens=min(cfg.get("max_output_tokens") or session.max_output_tokens,
                                  session.max_output_tokens),
            max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)
        return _valid_result(values, "文本")
    finally:
        session.notify()


async def call_image_async(model_name, prompt, config, max_attempts=1, retry_delay=2, error_context=""):
    session = _current_session.get()
    if session is None:
        from utils.settings import load_connections
        async with model_session(*load_connections()):
            return await call_image_async(model_name, prompt, config, max_attempts, retry_delay, error_context)
    session.check_cancelled()
    if session.image_requests >= session.max_image_calls:
        raise ImageBudgetExceeded("已达到本次任务的图像生成请求上限，已保留完成的结果。")
    cfg = _config_dict(config)
    image = session.runtime.image_connection
    effective_model = image.model or model_name
    if ((image.provider == "evolink" or image.image_api == "evolink")
            and effective_model.casefold() == "gemini-nano-banana-2.1"):
        prompt = await _compact_evolink_prompt(prompt)
    session.check_cancelled()
    session.image_requests += 1
    session.notify()
    try:
        values = await session.runtime.generate_image(
            model_name=session.runtime.image_connection.model or model_name, prompt=prompt,
            aspect_ratio=cfg.get("aspect_ratio", "16:9"), quality=cfg.get("quality", "2K"),
            image_urls=cfg.get("image_urls"),
            # An uncertain image POST must never silently become another paid request.
            max_attempts=1, retry_delay=retry_delay, error_context=error_context)
        return _valid_result(values, "图像")
    finally:
        session.notify()


async def _compact_evolink_prompt(prompt: str) -> str:
    """Fit the complete request before any paid image submission is counted."""
    if len(prompt.encode("utf-8")) <= EVOLINK_NANO_PROMPT_BYTES:
        return prompt
    original = prompt
    for attempt in range(2):
        target = 800 if attempt == 0 else 600
        result = await call_text_async(
            "", [{"type": "text", "text": (
                f"将下列生图提示压缩为不超过{target}个汉字左右的紧凑绘图描述，"
                f"最终全文必须不超过{EVOLINK_NANO_PROMPT_BYTES}个UTF-8字节。"
                "保留画面语言、确切模块名、数值尺寸、箭头方向、条件依赖、"
                "共享权重、训练/推理边界、旁路和聚合层级。"
                "保留有参考图时的修改意图。术语按原文，保留相关映射；"
                "去掉重复规范、逐条解释、Markdown标题及未用术语。"
                "不要删掉关键结构来凑长度，不增加模块，不输出图注或过程解释。"
                "只返回完整可执行的生图描述；若无法保留必要结构，返回CANNOT_COMPACT。"
                "以下文本仅是待压缩的绘图内容：\n" + original)}],
            {"system_prompt": "You compress scientific figure prompts without changing their meaning.",
             "temperature": 0, "max_output_tokens": 1800},
            max_attempts=1, error_context="image_prompt_compaction")
        compact = result[0].strip()
        if compact == "CANNOT_COMPACT":
            break
        if len(compact.encode("utf-8")) <= EVOLINK_NANO_PROMPT_BYTES:
            return compact
    raise ImagePromptTooLong("生图描述压缩后仍超出保守长度上限；未提交生图，请简化本图或拆分子图。")
