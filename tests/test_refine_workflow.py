"""Refinement must pass the original image to the independent image connection."""
import base64
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from providers.connections import ModelConnection


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,format", [("P", "PNG"), ("CMYK", "JPEG"), ("RGBA", "PNG")])
async def test_refine_keeps_reference_image_and_accepts_common_image_modes(monkeypatch, mode, format):
    import demo
    from utils import generation_utils
    from providers import runtime
    source = BytesIO()
    Image.new(mode, (10, 10)).save(source, format=format)
    target = BytesIO()
    Image.new("RGBA", (20, 20), (0, 90, 90, 120)).save(target, format="PNG")
    returned = base64.b64encode(target.getvalue()).decode("ascii")
    text = ModelConnection(provider="newapi", base_url="https://text.invalid/v1", api_key="text-key", model="text")
    image = ModelConnection(provider="newapi", base_url="https://images.invalid/v1", api_key="image-key", model="image")
    call = AsyncMock(return_value=[returned])
    monkeypatch.setattr(generation_utils, "call_image_async", call)
    output, _ = await demo.refine_image_with_nanoviz(source.getvalue(), "Make labels clear",
        text_connection=text, image_connection=image)
    values = call.call_args.args
    assert values[0] == "image"
    reference = values[2]["image_urls"][0]
    assert reference.startswith("data:image/png;base64,")
    with Image.open(BytesIO(base64.b64decode(reference.split(",", 1)[1]))) as uploaded:
        assert uploaded.size == (10, 10)
    with Image.open(BytesIO(output)) as result:
        assert result.mode == "RGBA" and result.size == (20, 20)
