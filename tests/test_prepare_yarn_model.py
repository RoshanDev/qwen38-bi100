import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_yarn_model.py"
SPEC = importlib.util.spec_from_file_location("prepare_yarn_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LayersBlockTypeTests(unittest.TestCase):
    def test_translates_qwen_hybrid_layout(self):
        text_config = {
            "num_hidden_layers": 8,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 2,
        }

        result = MODULE.build_layers_block_type(text_config)

        self.assertEqual(8, len(result))
        self.assertEqual(2, result.count("attention"))
        self.assertEqual(6, result.count("linear_attention"))

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "length does not match"):
            MODULE.build_layers_block_type(
                {
                    "num_hidden_layers": 4,
                    "layer_types": ["full_attention"],
                }
            )

    def test_rejects_unknown_layer_type(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported Qwen layer"):
            MODULE.build_layers_block_type(
                {
                    "num_hidden_layers": 2,
                    "layer_types": ["full_attention", "mystery"],
                }
            )


if __name__ == "__main__":
    unittest.main()
