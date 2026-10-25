import base64
import hashlib
import json
from io import BytesIO

import pytest
import yaml
from PIL import Image

from providers.connections import ModelConnection
from utils.results import final_image_key, evolution_stages, load_history, normalize_history, result_downloads, results_zip
from utils.settings import load_connections, save_connections


def encoded_png():
    stream = BytesIO()
    Image.new("RGBA", (12, 8), (20, 80, 200, 120)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_independent_connection_save_does_not_rewrite_legacy_config(tmp_path, monkeypatch):
    for name in ("TEXT_PROVIDER", "TEXT_API_KEY", "IMAGE_PROVIDER", "IMAGE_API_KEY", "GOOGLE_API_KEY", "EVOLINK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "configs").mkdir()
    legacy = tmp_path / "configs" / "model_config.yaml"
    legacy.write_text("evolink:\n  api_key: old-secret\n", encoding="utf-8")
    checksum = hashlib.sha256(legacy.read_bytes()).digest()
    text = ModelConnection(provider="newapi", base_url="https://newapi.omgteam.me/v1", api_key="text-secret", model="vision")
    image = ModelConnection(provider="openai", base_url="https://api.openai.com/v1", api_key="image-secret", model="image")
    save_connections(text, image, tmp_path)
    assert load_connections(tmp_path) == (text, image)
    assert hashlib.sha256(legacy.read_bytes()).digest() == checksum
    assert "secret" not in repr(text)
    assert "api_key" not in text.to_public_dict()


def test_changing_env_provider_does_not_forward_previous_platform_key(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "connections.local.yaml").write_text(yaml.safe_dump({"connections": {
        "text": {"provider": "evolink", "api_key": "evolink-secret", "model": "old"},
        "image": {"provider": "openai", "api_key": "image-secret", "model": "img"}}}), encoding="utf-8")
    monkeypatch.setenv("TEXT_PROVIDER", "newapi")
    monkeypatch.delenv("TEXT_API_KEY", raising=False)
    text, image = load_connections(tmp_path)
    assert text.provider == "newapi" and text.api_key == ""
    assert text.base_url == "https://newapi.omgteam.me/v1"
    assert image.api_key == "image-secret"


def test_fifth_round_selected_and_transparency_preserved_in_downloads():
    png = encoded_png()
    result = {"target_diagram_critic_desc0_base64_png": png,
              "target_diagram_critic_desc4_base64_png": png,
              "target_diagram_critic_desc4": "final",
              "eval_image_field": "target_diagram_critic_desc4_base64_png"}
    assert final_image_key(result).endswith("desc4_base64_png")
    assert evolution_stages(result)[-1]["name"] == "评审第 5 轮"
    with Image.open(BytesIO(result_downloads(result)["png"])) as image:
        assert image.mode == "RGBA"
    import zipfile
    with zipfile.ZipFile(BytesIO(results_zip([result]))) as archive:
        assert archive.namelist() == ["candidate_1.png"]


def test_history_rejects_file_outside_results_directory(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    outside = tmp_path / "other.json"
    outside.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_history(outside, root)
    old = root / "old.json"
    old.write_text(json.dumps([{"caption": "legacy"}]), encoding="utf-8")
    assert load_history(old, root)["results"] == [{"caption": "legacy"}]


@pytest.mark.parametrize("field,value", [("settings", []), ("settings", None), ("settings", "bad"),
    ("inputs", {}), ("inputs", ["bad"]), ("usage", None), ("stages", []), ("status", [])])
def test_invalid_history_metadata_is_rejected_before_entering_ui_state(field, value):
    with pytest.raises(ValueError):
        normalize_history({"results": [], field: value})


@pytest.mark.parametrize("source", ["connections: bad", "connections:\n  text: bad", "api_keys: bad", "connections: [unclosed"])
def test_invalid_connection_yaml_is_a_readable_configuration_error(tmp_path, source):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "model_config.yaml").write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        load_connections(tmp_path)


def test_zip_uses_stable_candidate_numbers_despite_completion_order():
    import zipfile
    png = encoded_png()
    values = [{"candidate_id": index, "target_diagram_desc0_base64_png": png} for index in (4, 0)]
    with zipfile.ZipFile(BytesIO(results_zip(values))) as archive:
        assert archive.namelist() == ["candidate_5.png", "candidate_1.png"]


def test_cmyk_image_can_be_downloaded_as_png():
    from utils.results import png_bytes
    source = BytesIO()
    Image.new("CMYK", (10, 10)).save(source, format="JPEG")
    data = png_bytes(base64.b64encode(source.getvalue()).decode("ascii"))
    with Image.open(BytesIO(data)) as output:
        assert output.format == "PNG" and output.mode == "RGB"
