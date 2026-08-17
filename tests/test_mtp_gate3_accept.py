import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mtp_gate3_accept.py"
SPEC = importlib.util.spec_from_file_location("mtp_gate3_accept", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MtpGate3Tests(unittest.TestCase):
    def test_k1_accepts_matching_draft(self):
        result = MODULE.k1_step(7, 7)
        self.assertTrue(result["accepted_draft"])
        self.assertEqual(result["next_token"], 7)

    def test_k1_rejects_mismatch_and_uses_target(self):
        result = MODULE.k1_step(7, 9)
        self.assertFalse(result["accepted_draft"])
        self.assertEqual(result["next_token"], 9)

    def test_greedy_prefix(self):
        result = MODULE.accept_reject_greedy([1, 2, 3], [1, 2, 9])
        self.assertEqual(result["accepted"], 2)

    def test_k1_rounds_reject_falls_back_to_target(self):
        result = MODULE.k1_rounds([7, 8], [9, 10])
        self.assertFalse(result["all_accepted"])
        self.assertEqual(result["emitted_ids"], [9, 10])
        self.assertTrue(result["matches_target"])

    def test_k1_rounds_all_accept_matches_target(self):
        result = MODULE.k1_rounds([1, 2], [1, 2])
        self.assertTrue(result["all_accepted"])
        self.assertEqual(result["emitted_ids"], [1, 2])

    def test_pipeline_always_emits_target_t0(self):
        result = MODULE.k1_pipeline(5, [9, 8], [7, 8])
        self.assertEqual(result["emitted_ids"][0], 5)
        self.assertTrue(result["t0_identity"])
        self.assertEqual(result["emitted_ids"], [5, 7, 8])
        self.assertTrue(result["matches_no_mtp_prefix"])

    def test_pipeline_accepts_matching_drafts(self):
        result = MODULE.k1_pipeline(5, [7, 8], [7, 8])
        self.assertTrue(result["all_accepted"])
        self.assertEqual(result["emitted_ids"], [5, 7, 8])


REPLAY = Path(__file__).resolve().parents[1] / "scripts" / "mtp_gate3_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("mtp_gate3_replay", REPLAY)
REPLAY_MOD = importlib.util.module_from_spec(REPLAY_SPEC)
assert REPLAY_SPEC.loader is not None
sys.modules[REPLAY_SPEC.name] = REPLAY_MOD
REPLAY_SPEC.loader.exec_module(REPLAY_MOD)


class MtpGate3ReplaySelectTests(unittest.TestCase):
    def test_select_prefers_saved_vector(self):
        class _Vec:
            def dim(self):
                return 1

        hidden, how = REPLAY_MOD.select_hidden_row({"hidden": _Vec(), "shape": [21, 5120], "rows": 21})
        self.assertEqual(how, "saved_vector")
        self.assertTrue(hasattr(hidden, "dim"))


if __name__ == "__main__":
    unittest.main()
