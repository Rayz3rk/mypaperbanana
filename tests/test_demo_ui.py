from streamlit.testing.v1 import AppTest
from pathlib import Path

from providers.connections import ModelConnection


def test_ui_has_independent_keys_and_switching_platform_clears_only_its_key(monkeypatch):
    from utils import settings
    text = ModelConnection(provider="newapi", base_url="https://newapi.omgteam.me/v1", api_key="text-private", model="vision")
    image = ModelConnection(provider="openai", base_url="https://api.openai.com/v1", api_key="image-private", model="image")
    monkeypatch.setattr(settings, "load_connections", lambda: (text, image))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo.py")).run(timeout=45)
    assert not app.exception
    assert app.text_input(key="text_api_key").value == "text-private"
    assert app.text_input(key="image_api_key").value == "image-private"
    app.selectbox(key="text_provider").set_value("deepseek").run(timeout=45)
    assert not app.exception
    assert app.text_input(key="text_api_key").value == ""
    assert app.text_input(key="image_api_key").value == "image-private"
    assert app.text_input(key="text_base_url").value == "https://api.deepseek.com/v1"
    assert any(button.label == "精修 / 生成图片" for button in app.button)


def test_running_job_shows_completed_candidates_and_cancel_control(monkeypatch):
    import base64
    from io import BytesIO
    from PIL import Image
    from utils import settings
    connection = ModelConnection(provider="newapi", base_url="https://newapi.omgteam.me/v1", api_key="placeholder", model="test")
    monkeypatch.setattr(settings, "load_connections", lambda: (connection, connection))
    stream = BytesIO()
    Image.new("RGB", (12, 8)).save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    class RunningJob:
        id = "local-test-job"
        cancelled = False
        def is_alive(self): return True
        def cancel(self): self.cancelled = True
        def snapshot(self):
            return {"status": "running", "total": 6, "usage": {}, "settings": {"exp_mode": "demo_planner_critic"},
                    "results": [{"candidate_id": 4, "status": "completed", "target_diagram_desc0_base64_png": encoded}],
                    "stages": {"4": "critic_3"}}
    job = RunningJob()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo.py"))
    app.session_state["active_job"] = job
    app.run(timeout=45)
    assert not app.exception
    assert any(item.value == "**候选 5**" for item in app.markdown)
    cancel = next(button for button in app.button if button.label == "取消任务并保留已完成结果")
    cancel.click().run(timeout=45)
    assert job.cancelled
    assert not app.exception
