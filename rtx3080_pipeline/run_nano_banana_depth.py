"""Request a Nano Banana depth-style image from Gemini Interactions API.

The API key is read from GEMINI_API_KEY. Do not hardcode credentials here.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = """You are doing a computer-vision depth estimation task.

Input: one RGB render of a scene.
Output: only one image, no text, no labels, no legend, no axes, no colorbar.

Generate a dense monocular depth map for the exact same camera view and exact same image crop.
Preserve object boundaries and pixel geometry as closely as possible.
Use a single grayscale channel:
- white means closer to the camera
- black means farther from the camera

Do not stylize the result. Do not redraw the RGB scene. Return only the depth map image."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3-pro-image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def find_image_blocks(value: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(value, dict):
        block_type = value.get("type") or value.get("mime_type") or value.get("mimeType")
        if isinstance(block_type, str) and "image" in block_type.lower() and ("data" in value or "bytesBase64Encoded" in value):
            blocks.append(value)
        for key in ("output_image", "outputImage", "image", "inlineData"):
            nested = value.get(key)
            if isinstance(nested, dict):
                blocks.extend(find_image_blocks(nested))
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                blocks.extend(find_image_blocks(nested))
    elif isinstance(value, list):
        for item in value:
            blocks.extend(find_image_blocks(item))
    return blocks


def redact_base64(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key in {"data", "bytesBase64Encoded"} and isinstance(child, str) and len(child) > 256:
                out[key] = f"<base64 {len(child)} chars>"
            else:
                out[key] = redact_base64(child)
        return out
    if isinstance(value, list):
        return [redact_base64(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    image_data = base64.b64encode(args.input_image.read_bytes()).decode("utf-8")
    payload = {
        "model": args.model,
        "input": [
            {"type": "text", "text": args.prompt},
            {"type": "image", "mime_type": image_mime(args.input_image), "data": image_data},
        ],
    }
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {error}", file=sys.stderr)
        raise

    data = json.loads(raw)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    (args.output_image.with_suffix(".response_redacted.json")).write_text(
        json.dumps(redact_base64(data), indent=2) + "\n",
        encoding="utf-8",
    )

    image_blocks = find_image_blocks(data)
    if not image_blocks:
        raise RuntimeError(f"No image block found. Redacted response saved to {args.output_image.with_suffix('.response_redacted.json')}")

    block = image_blocks[-1]
    encoded = block.get("data") or block.get("bytesBase64Encoded")
    if not isinstance(encoded, str):
        raise RuntimeError("Image block did not contain base64 data")
    args.output_image.write_bytes(base64.b64decode(encoded))
    print(f"Saved Nano Banana output to {args.output_image}")


if __name__ == "__main__":
    main()
