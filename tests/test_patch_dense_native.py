import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_dense_native.py"
SPEC = importlib.util.spec_from_file_location("patch_dense_native", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DenseNativePatcherTests(unittest.TestCase):
    def test_replaces_each_expected_block_once(self):
        source = "\n".join(
            [MODULE.LOOP1_OLD, MODULE.CHUNK_OLD, MODULE.RMS_OLD]
        )

        result = MODULE.patch_source(source)

        self.assertIn("QWEN38_LOOP1_NATIVE", result)
        self.assertIn("QWEN38_CHUNK_PARALLEL", result)
        self.assertIn("QWEN38_RMSNORM_NATIVE", result)

    def test_refuses_already_patched_or_unexpected_source(self):
        with self.assertRaisesRegex(RuntimeError, "expected one loop1"):
            MODULE.patch_source("not the expected adapter")


if __name__ == "__main__":
    unittest.main()
