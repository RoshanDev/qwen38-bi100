import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bridge.images import (
    ImageDescriber,
    decode_data_url,
    image_url_from_item,
    ocr_image,
    render_image_block,
)
from bridge.responses_to_chat import responses_request_to_chat


def _text_png(text: str) -> bytes:
    image = Image.new("RGB", (720, 160), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.text((24, 48), text, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ImageBridgeTests(unittest.TestCase):
    def test_data_url_round_trip(self) -> None:
        payload = b"png-bytes"
        encoded = base64.b64encode(payload).decode("ascii")
        data, mime = decode_data_url(f"data:image/png;base64,{encoded}")
        self.assertEqual(b"png-bytes", data)
        self.assertEqual("image/png", mime)

    def test_nested_image_url_object(self) -> None:
        url = image_url_from_item(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}}
        )
        self.assertEqual("data:image/png;base64,QQ==", url)

    def test_request_uses_image_describer(self) -> None:
        payload = responses_request_to_chat(
            {
                "model": "Qwen3.8-27B",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "看图"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,QQ==",
                            },
                        ],
                    }
                ],
            },
            describe_image=lambda item: f"DESCRIBED:{item['image_url']}",
        )
        user = payload["messages"][-1]["content"]
        self.assertIn("看图", user)
        self.assertIn("DESCRIBED:data:image/png;base64,QQ==", user)
        self.assertNotIn("text-only", user)

    def test_top_level_image_item(self) -> None:
        payload = responses_request_to_chat(
            {
                "input": [
                    {"type": "input_image", "image_url": "/tmp/shot.png"},
                ]
            },
            describe_image=lambda item: f"FILE:{item['image_url']}",
        )
        self.assertEqual("FILE:/tmp/shot.png", payload["messages"][-1]["content"])

    def test_ocr_reads_generated_screenshot(self) -> None:
        data = _text_png("CODEX_VISION_OK")
        text = ocr_image(data, languages="eng", timeout_seconds=20)
        self.assertIn("CODEX_VISION_OK", text.replace(" ", ""))

    def test_local_file_ocr_via_describer(self) -> None:
        data = _text_png("CODEX_VISION_OK")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shot.png"
            path.write_bytes(data)
            describer = ImageDescriber(ocr_languages="eng", timeout_seconds=20)
            result = describer(
                {"type": "input_image", "image_url": str(path.resolve())}
            )
        self.assertIn("[pasted image 1]", result)
        self.assertIn("CODEX_VISION_OK", result.replace(" ", ""))

    def test_render_block_includes_metadata(self) -> None:
        block = render_image_block(2, {"format": "PNG", "size": "10x10"}, "hello")
        self.assertEqual("[pasted image 2] format=PNG size=10x10\nhello", block)

    def test_catalog_declares_local_model(self) -> None:
        catalog = json.loads(
            Path("codex/model-catalog.json").read_text(encoding="utf-8")
        )
        model = catalog["models"][0]
        self.assertEqual("Qwen3.8-27B", model["slug"])
        self.assertEqual(400000, model["context_window"])
        self.assertIn("image", model["input_modalities"])
        self.assertTrue(
            model.get("base_instructions")
            or (model.get("model_messages") or {}).get("instructions_template")
        )


if __name__ == "__main__":
    unittest.main()
