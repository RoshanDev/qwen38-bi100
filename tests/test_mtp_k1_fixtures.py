import json
import unittest
from pathlib import Path

FIX = Path(__file__).resolve().parent / "fixtures" / "mtp_k1"


class MtpK1FixtureTests(unittest.TestCase):
    def test_accept_fixture(self):
        data = json.loads((FIX / "accept.json").read_text(encoding="utf-8"))
        self.assertTrue(data["accepted"])
        self.assertEqual(data["draft1"], data["t1"])
        self.assertEqual(data["t0"], 32)
        self.assertEqual(data["t1"], 248046)
        self.assertEqual(data["rms"], "gemma_one_plus_weight")

    def test_reject_fixture(self):
        data = json.loads((FIX / "reject.json").read_text(encoding="utf-8"))
        self.assertFalse(data["accepted"])
        self.assertNotEqual(data["draft1"], data["t1"])
        self.assertEqual(data["emitted"], [32, 248046])


if __name__ == "__main__":
    unittest.main()
