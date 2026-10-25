"""The launcher must preserve listeners and detect changed dependencies."""

import socket

import pytest

from scripts.launcher_utils import find_available_port, requirements_fingerprint


def test_requirements_fingerprint_changes_with_content(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("streamlit>=1.40,<2\n", encoding="utf-8")
    original = requirements_fingerprint(requirements)
    assert requirements_fingerprint(requirements) == original
    requirements.write_text("streamlit>=1.50,<2\n", encoding="utf-8")
    assert requirements_fingerprint(requirements) != original


def test_occupied_port_is_skipped_and_listener_remains_alive():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied = listener.getsockname()[1]
        available = find_available_port(occupied, min(occupied + 30, 65536))
        assert available != occupied
        with socket.create_connection(("127.0.0.1", occupied), timeout=1):
            connection, _ = listener.accept()
            connection.close()


def test_exhausted_port_range_fails_instead_of_killing_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="No available port"):
            find_available_port(occupied, occupied + 1)


@pytest.mark.parametrize("start, stop", [(0, 10), (8501, 8501), (1, 65537)])
def test_invalid_port_ranges_are_rejected(start, stop):
    with pytest.raises(ValueError, match="Port range"):
        find_available_port(start, stop)
