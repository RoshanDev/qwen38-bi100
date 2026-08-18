import math
import unittest

from scripts.mtp_k1.replay_stats import (
    DECISION_ALIGN,
    DECISION_GPU_MAYBE,
    DECISION_GPU_SPEC,
    DECISION_STOP,
    DIAGNOSTIC_POSITION_OFFSETS,
    OFFICIAL_POSITION_OFFSETS,
    attention_weights,
    decide_from_top1,
    is_official_offset,
    softmax,
    step_stats,
    summarize_steps,
)


class SoftmaxAndAttentionTests(unittest.TestCase):
    def test_single_key_is_one_by_one(self):
        weights = attention_weights([1.0, 0.0], [[0.5, 0.25]], scale=1.0)
        self.assertEqual(len(weights), 1)
        self.assertTrue(math.isclose(weights[0], 1.0, abs_tol=1e-12))

    def test_softmax_one_element_is_always_one(self):
        self.assertTrue(math.isclose(softmax([-12.0])[0], 1.0, abs_tol=1e-12))

    def test_two_keys_are_not_one_by_one(self):
        weights = attention_weights([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], scale=1.0)
        self.assertEqual(len(weights), 2)
        self.assertTrue(math.isclose(sum(weights), 1.0, abs_tol=1e-9))
        self.assertGreater(weights[0], weights[1])
        self.assertLess(weights[0], 1.0)

    def test_empty_keys_fail_loud(self):
        with self.assertRaises(ValueError):
            attention_weights([1.0], [], scale=1.0)


class ReplayStatsTests(unittest.TestCase):
    def test_step_stats_top1_and_rank(self):
        logits = [0.1, 3.0, 2.5, 0.2]
        stats = step_stats(logits, target_id=2, top_k=3)
        self.assertEqual(stats["draft_id"], 1)
        self.assertFalse(stats["top1_hit"])
        self.assertTrue(stats["top5_hit"])
        self.assertEqual(stats["rank"], 2)
        self.assertTrue(math.isclose(stats["logit_margin_draft_minus_target"], 0.5, abs_tol=1e-12))

    def test_step_stats_exact_top1(self):
        stats = step_stats([1.0, 9.0, 2.0], target_id=1)
        self.assertTrue(stats["top1_hit"])
        self.assertEqual(stats["rank"], 1)
        self.assertTrue(math.isclose(stats["logit_margin_draft_minus_target"], 0.0, abs_tol=1e-12))

    def test_summarize_rates(self):
        rows = [
            step_stats([5.0, 1.0, 0.0], 0),
            step_stats([1.0, 5.0, 4.0], 2),
        ]
        summary = summarize_steps(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["top1_hits"], 1)
        self.assertEqual(summary["top5_hits"], 2)
        self.assertTrue(math.isclose(summary["top1_rate"], 0.5, abs_tol=1e-12))
        self.assertTrue(math.isclose(summary["mean_rank"], 1.5, abs_tol=1e-12))


class DecisionTableTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(decide_from_top1(0.145)["decision"], DECISION_STOP)
        self.assertEqual(decide_from_top1(0.2999)["decision"], DECISION_STOP)
        self.assertEqual(decide_from_top1(0.30)["decision"], DECISION_ALIGN)
        self.assertEqual(decide_from_top1(0.599)["decision"], DECISION_ALIGN)
        self.assertEqual(decide_from_top1(0.60)["decision"], DECISION_GPU_MAYBE)
        self.assertEqual(decide_from_top1(0.75)["decision"], DECISION_GPU_SPEC)

    def test_position_offsets_are_diagnostic_only(self):
        self.assertEqual(OFFICIAL_POSITION_OFFSETS, (0,))
        self.assertEqual(DIAGNOSTIC_POSITION_OFFSETS, (-2, -1, 1, 2))
        self.assertTrue(is_official_offset(0))
        self.assertFalse(is_official_offset(1))
        self.assertFalse(is_official_offset(-1))


class AssembleRecordTests(unittest.TestCase):
    def test_uses_recorded_position_not_counter(self):
        from scripts.mtp_k1.trace import assemble_records, classify_dump

        self.assertEqual(classify_dump(2048), "skip")
        self.assertEqual(classify_dump(21), "prefill")
        self.assertEqual(classify_dump(1), "decode")
        assembled = assemble_records(
            [32, 248046, 7],
            [
                {"kind": "skip", "rows": 2048},
                {"kind": "prefill", "rows": 21, "position": 20, "how": "selected"},
                {"kind": "decode", "rows": 1, "position": 21, "how": "last"},
            ],
        )
        self.assertEqual(assembled["n_steps"], 2)
        self.assertEqual(assembled["steps"][0]["position"], 20)
        self.assertEqual(assembled["steps"][0]["sampled_token"], 32)
        self.assertEqual(assembled["steps"][0]["next_token"], 248046)
        self.assertEqual(assembled["steps"][1]["position"], 21)
        self.assertEqual(assembled["steps"][1]["next_token"], 7)

    def test_missing_position_fails_loud(self):
        from scripts.mtp_k1.trace import assemble_records

        with self.assertRaises(ValueError):
            assemble_records([1, 2], [{"kind": "prefill", "rows": 8}])


class TorchCacheTests(unittest.TestCase):
    def test_append_grows_time_axis(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from scripts.mtp_k1.kv import MTPKVCache, attend_decode

        cache = MTPKVCache()
        k1 = torch.zeros(1, 2, 1, 4)
        v1 = torch.ones(1, 2, 1, 4)
        k2 = torch.full((1, 2, 1, 4), 2.0)
        v2 = torch.full((1, 2, 1, 4), 3.0)
        full_k, full_v = cache.append(k1, v1)
        self.assertEqual(len(cache), 1)
        self.assertEqual(tuple(full_k.shape), (1, 2, 1, 4))
        full_k, full_v = cache.append(k2, v2)
        self.assertEqual(len(cache), 2)
        self.assertEqual(tuple(full_k.shape), (1, 2, 2, 4))
        self.assertEqual(tuple(full_v.shape), (1, 2, 2, 4))
        q = torch.ones(1, 2, 1, 4)
        ctx = attend_decode(q, full_k.repeat_interleave(3, dim=1), full_v.repeat_interleave(3, dim=1), 4)
        self.assertEqual(tuple(ctx.shape), (1, 6, 1, 4))


if __name__ == "__main__":
    unittest.main()
