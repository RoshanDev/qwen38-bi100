"""Convert Codex image attachments into compact text for a text-only model."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlparse


DescribeImage = Callable[[dict[str, Any]], str]


def image_url_from_item(item: dict[str, Any]) -> str:
    value = item.get("image_url", item.get("image"))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("url", "image_url", "file_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def decode_data_url(url: str) -> tuple[bytes, str]:
    header, separator, payload = url.partition(",")
    if not separator:
        raise ValueError("data URL is missing payload")
    mime = "application/octet-stream"
    if header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    return base64.b64decode(payload), mime


def _read_local_path(path_value: str) -> bytes:
    path = Path(unquote(path_value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    return path.read_bytes()


def load_image_bytes(url: str, *, timeout_seconds: int, use_system_proxy: bool) -> tuple[bytes, str]:
    if url.startswith("data:"):
        return decode_data_url(url)
    if url.startswith("file://"):
        parsed = urlparse(url)
        return _read_local_path(parsed.path), "application/octet-stream"
    local = Path(url).expanduser()
    if local.is_file():
        return local.read_bytes(), "application/octet-stream"
    if url.startswith(("http://", "https://")):
        request = urllib_request.Request(url, method="GET")
        opener = (
            urllib_request.build_opener()
            if use_system_proxy
            else urllib_request.build_opener(urllib_request.ProxyHandler({}))
        )
        with opener.open(request, timeout=timeout_seconds) as response:
            mime = response.headers.get_content_type() or "application/octet-stream"
            return response.read(), mime
    raise ValueError(f"unsupported image reference: {url[:80]}")


def image_metadata(data: bytes) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {"bytes": len(data)}

    with Image.open(io.BytesIO(data)) as image:
        return {
            "format": image.format or "unknown",
            "size": f"{image.width}x{image.height}",
            "mode": image.mode,
            "bytes": len(data),
        }


def ocr_image(data: bytes, *, languages: str, timeout_seconds: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(data)
        temp_path = handle.name
    try:
        completed = subprocess.run(
            [
                "tesseract",
                temp_path,
                "stdout",
                "-l",
                languages,
                "--psm",
                "6",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"(ocr unavailable: {exc})"
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "tesseract failed").strip()
        return f"(ocr failed: {error})"
    return completed.stdout.strip()


def _vision_chat_payload(model: str, image_url: str) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 800,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this screenshot for a coding agent. "
                            "Transcribe visible text exactly. "
                            "Mention UI layout, errors, and anything a developer would need."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }


def describe_with_vision_api(
    image_url: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: int,
    use_system_proxy: bool,
) -> str:
    payload = _vision_chat_payload(model, image_url)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib_request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = (
        urllib_request.build_opener()
        if use_system_proxy
        else urllib_request.build_opener(urllib_request.ProxyHandler({}))
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            body = json.load(response)
    except (urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vision API failed: {exc}") from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("vision API returned no choices")
    message = choices[0].get("message", {})
    text = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("vision API returned empty content")
    return text.strip()


def render_image_block(index: int, metadata: dict[str, Any], body: str) -> str:
    details = " ".join(
        f"{key}={value}"
        for key, value in metadata.items()
        if key != "bytes" and value not in {None, ""}
    )
    header = f"[pasted image {index}]"
    if details:
        header = f"{header} {details}"
    body = body.strip() or "(no readable text found)"
    return f"{header}\n{body}"


class ImageDescriber:
    def __init__(
        self,
        *,
        vision_base_url: str = "",
        vision_model: str = "Qwen3.8-27B",
        vision_api_key: str = "",
        timeout_seconds: int = 60,
        ocr_languages: str = "chi_sim+eng",
        use_system_proxy: bool = False,
    ) -> None:
        self.vision_base_url = vision_base_url.rstrip("/")
        self.vision_model = vision_model
        self.vision_api_key = vision_api_key
        self.timeout_seconds = timeout_seconds
        self.ocr_languages = ocr_languages
        self.use_system_proxy = use_system_proxy
        self._index = 0

    def __call__(self, item: dict[str, Any]) -> str:
        self._index += 1
        url = image_url_from_item(item)
        if not url:
            return f"[pasted image {self._index}]\n(missing image_url)"

        if self.vision_base_url:
            api_url = url
            if not api_url.startswith(("data:", "http://", "https://")):
                try:
                    data, mime = load_image_bytes(
                        url,
                        timeout_seconds=self.timeout_seconds,
                        use_system_proxy=self.use_system_proxy,
                    )
                    encoded = base64.b64encode(data).decode("ascii")
                    api_url = f"data:{mime};base64,{encoded}"
                except (OSError, ValueError) as exc:
                    return f"[pasted image {self._index}]\n(failed to load image: {exc})"
            try:
                text = describe_with_vision_api(
                    api_url,
                    base_url=self.vision_base_url,
                    model=self.vision_model,
                    api_key=self.vision_api_key,
                    timeout_seconds=self.timeout_seconds,
                    use_system_proxy=self.use_system_proxy,
                )
                return render_image_block(self._index, {}, text)
            except RuntimeError as exc:
                return f"[pasted image {self._index}]\n({exc})"

        try:
            data, _mime = load_image_bytes(
                url,
                timeout_seconds=self.timeout_seconds,
                use_system_proxy=self.use_system_proxy,
            )
        except (OSError, ValueError, urllib_error.URLError) as exc:
            return f"[pasted image {self._index}]\n(failed to load image: {exc})"

        metadata = image_metadata(data)
        text = ocr_image(
            data,
            languages=self.ocr_languages,
            timeout_seconds=self.timeout_seconds,
        )
        return render_image_block(self._index, metadata, text)


def describer_from_env() -> ImageDescriber:
    return ImageDescriber(
        vision_base_url=os.environ.get("QWEN_VISION_BASE_URL", "").strip(),
        vision_model=os.environ.get("QWEN_VISION_MODEL", "Qwen3.8-27B"),
        vision_api_key=os.environ.get("QWEN_VISION_API_KEY", ""),
        timeout_seconds=int(os.environ.get("QWEN_VISION_TIMEOUT_SECONDS", "60")),
        ocr_languages=os.environ.get("QWEN_VISION_OCR_LANGS", "chi_sim+eng"),
        use_system_proxy=os.environ.get("QWEN_UPSTREAM_USE_SYSTEM_PROXY", "0").lower()
        in {"1", "true", "yes", "on"},
    )
