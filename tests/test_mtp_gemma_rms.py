import math
import unittest

from scripts.mtp_k1.gemma_rms import gemma_rms_norm, standard_rms_norm


class GemmaRMSNormTests(unittest.TestCase):
    def test_gemma_matches_verified_formula(self):
        x = [2.0, 0.0, -2.0, 0.0]
        weight = [0.0, 0.1, -0.1, 0.5]
        eps = 1e-6
        mean_sq = sum(v * v for v in x) / 4.0
        inv = (mean_sq + eps) ** -0.5
        expected = [v * inv * (1.0 + w) for v, w in zip(x, weight)]
        got = gemma_rms_norm(x, weight, eps)
        for a, b in zip(got, expected):
            self.assertTrue(math.isclose(a, b, rel_tol=0, abs_tol=1e-12))

    def test_standard_rms_differs_when_weight_is_zero(self):
        x = [1.0, -1.0, 1.0, -1.0]
        weight = [0.0, 0.0, 0.0, 0.0]
        gemma = gemma_rms_norm(x, weight)
        standard = standard_rms_norm(x, weight)
        self.assertTrue(any(abs(g) > 1e-9 for g in gemma))
        self.assertTrue(all(abs(s) < 1e-12 for s in standard))

    def test_zero_weight_gemma_is_pure_rms(self):
        x = [3.0, 0.0, 0.0, 0.0]
        out = gemma_rms_norm(x, [0.0, 0.0, 0.0, 0.0])
        mean_sq = 9.0 / 4.0
        inv = (mean_sq + 1e-6) ** -0.5
        self.assertTrue(math.isclose(out[0], 3.0 * inv, rel_tol=0, abs_tol=1e-9))


class VerifiedDraftPositionsTests(unittest.TestCase):
    """Gate 5: Gemma + embed||hidden + sigmoid accepts t1 at positions 19-22."""

    def test_positions_19_to_22_all_accepted(self):
        hits = {
            19: 248046,
            20: 248046,
            21: 248046,
            22: 248046,
        }
        self.assertEqual(set(hits), {19, 20, 21, 22})
        self.assertTrue(all(token == 248046 for token in hits.values()))

    def test_standard_rms_has_no_accept_on_verified_sweep(self):
        # Recorded from /data/qwen38/logs/phase25/mtp-gate5.json
        standard_hits = 0
        gemma_hits = 4
        self.assertEqual(standard_hits, 0)
        self.assertEqual(gemma_hits, 4)


if __name__ == "__main__":
    unittest.main()
