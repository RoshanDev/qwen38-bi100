import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mtp_gate2_logits.py"
SPEC = importlib.util.spec_from_file_location("mtp_gate2_logits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MtpGate2Tests(unittest.TestCase):
    def test_rotary_dim_quarter_head(self):
        self.assertEqual(MODULE.rotary_dim(256, 0.25), 64)

    def test_rotary_dim_rejects_odd(self):
        with self.assertRaises(ValueError):
            MODULE.rotary_dim(255, 0.25)

    def test_rms_norm_is_gemma_style(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed on the workstation")
        x = torch.ones(2, 4)
        weight = torch.zeros(4)
        out = MODULE._rms_norm(x, weight, 1e-6)
        self.assertTrue(torch.allclose(out, torch.ones(2, 4), atol=1e-5))

    def test_weight_keys_are_untied(self):
        self.assertEqual(MODULE.EMBED_KEY, "model.language_model.embed_tokens.weight")
        self.assertEqual(MODULE.LM_HEAD_KEY, "lm_head.weight")
        self.assertEqual(len(MODULE.EXPECTED_KEYS), 15)


if __name__ == "__main__":
    unittest.main()
