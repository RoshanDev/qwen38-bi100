import unittest
from pathlib import Path

from scripts.patch_qwen3_5_mtp import patch_source


class PatchQwen35MtpTests(unittest.TestCase):
    def test_patches_production_adapter_fixture(self):
        fixture = Path("/tmp/qwen3_5_prod.py")
        if not fixture.exists():
            self.skipTest("production adapter fixture not copied")
        out = patch_source(fixture.read_text(encoding="utf-8"))
        self.assertIn("last_hidden_states", out)
        self.assertIn("runtime_hook", out)
        self.assertIn("after_final_norm", out)
        self.assertEqual(out.count("def sample("), 1)


if __name__ == "__main__":
    unittest.main()
