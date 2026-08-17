import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mtp_gate1_load.py"
SPEC = importlib.util.spec_from_file_location("mtp_gate1_load", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MtpGate1Tests(unittest.TestCase):
    def test_expected_key_count_is_15(self):
        self.assertEqual(len(MODULE.EXPECTED_KEYS), 15)
        self.assertEqual(len(set(MODULE.EXPECTED_KEYS)), 15)

    def test_qwen38_shapes_match_gated_attention(self):
        shapes = MODULE.expected_shapes(5120, 17408, 24, 4, 256)
        self.assertEqual(shapes["mtp.fc.weight"], (5120, 10240))
        self.assertEqual(shapes["mtp.layers.0.self_attn.q_proj.weight"], (12288, 5120))
        self.assertEqual(shapes["mtp.layers.0.self_attn.k_proj.weight"], (1024, 5120))
        self.assertEqual(shapes["mtp.layers.0.self_attn.o_proj.weight"], (5120, 6144))
        self.assertEqual(set(shapes), set(MODULE.EXPECTED_KEYS))


if __name__ == "__main__":
    unittest.main()
