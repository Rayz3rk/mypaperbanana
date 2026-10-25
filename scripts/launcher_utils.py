"""Dependency-free helpers shared by the desktop launch scripts."""

from __future__ import annotations

import argparse
import hashlib
import socket
import sys
from pathlib import Path


def requirements_fingerprint(path: Path) -> str:
    """Invalidate installation markers when requirements or Python change."""
    digest = hashlib.sha256()
    digest.update(f"{sys.version_info.major}.{sys.version_info.minor}\n".encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def find_available_port(start: int = 8501, stop: int = 8600) -> int:
    """Find a free loopback port without terminating other processes."""
    if not 1 <= start < stop <= 65536:
        raise ValueError("Port range must satisfy 1 <= start < stop <= 65536")
    for port in range(start, stop):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No available port in {start}-{stop - 1}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("requirements", type=Path)
    port = commands.add_parser("port")
    port.add_argument("--start", type=int, default=8501)
    port.add_argument("--stop", type=int, default=8600)
    args = parser.parse_args()
    try:
        if args.command == "fingerprint":
            print(requirements_fingerprint(args.requirements))
        else:
            print(find_available_port(args.start, args.stop))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Launcher: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
