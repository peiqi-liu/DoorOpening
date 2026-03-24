#!/usr/bin/env python3
"""Serve a saved .viser recording with a local static Viser client.

This follows the official Viser hosting flow for recorded scenes saved with
`StateSerializer.serialize()`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.parse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import which


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a saved .viser recording for browser playback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recording", type=Path, help="Path to a .viser recording file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind the local HTTP server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local HTTP server.")
    parser.add_argument(
        "--client-dir",
        type=Path,
        default=None,
        help="Directory containing the built static Viser client. If missing, one will be created.",
    )
    parser.add_argument(
        "--force-rebuild-client",
        action="store_true",
        help="Rebuild the static Viser client even if it already exists.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the playback URL in the default browser after the server starts.",
    )
    return parser.parse_args()


def _build_client_if_needed(client_dir: Path, force_rebuild: bool) -> None:
    if client_dir.exists() and (client_dir / "index.html").exists() and not force_rebuild:
        return

    if client_dir.exists():
        shutil.rmtree(client_dir)

    build_cmd = which("viser-build-client")
    if build_cmd is None:
        raise SystemExit(
            "Could not find 'viser-build-client'. Install viser first, for example with `pip install viser`."
        )

    print(f"Building static Viser client in {client_dir}...")
    build_variants = [
        [build_cmd, "--out-dir", str(client_dir)],
        [build_cmd, "--output-dir", str(client_dir)],
    ]
    last_error = None
    for cmd in build_variants:
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error


def _relative_to_root(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


class _PlaybackHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, playback_path: str, **kwargs):
        self._playback_path = playback_path
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        if self.path in {"/", ""}:
            self.send_response(302)
            self.send_header("Location", self._playback_path)
            self.end_headers()
            return
        super().do_GET()


def main() -> None:
    args = _parse_args()
    recording = args.recording.expanduser().resolve()
    if not recording.exists():
        raise SystemExit(f"Recording does not exist: {recording}")
    if recording.suffix != ".viser":
        raise SystemExit(f"Expected a .viser file, got: {recording}")

    client_dir = args.client_dir.expanduser().resolve() if args.client_dir is not None else recording.parent / "_viser_client"
    _build_client_if_needed(client_dir, args.force_rebuild_client)

    serve_root = Path(os.path.commonpath([recording, client_dir])).resolve()
    client_rel = _relative_to_root(client_dir, serve_root)
    recording_rel = _relative_to_root(recording, serve_root)

    playback_path = "/" + urllib.parse.quote(recording_rel)
    client_url_path = f"/{client_rel.rstrip('/')}/?playbackPath={playback_path}"

    handler = partial(_PlaybackHandler, directory=str(serve_root), playback_path=client_url_path)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    browser_url = f"http://{browser_host}:{args.port}{client_url_path}"

    print(f"Serving Viser playback root: {serve_root}")
    print(f"Playback URL: {browser_url}")
    if args.host == "0.0.0.0":
        print(f"For SSH use: ssh -L {args.port}:127.0.0.1:{args.port} <remote-host>")

    if args.open_browser:
        webbrowser.open(browser_url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down playback server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Failed to build the static Viser client: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)
