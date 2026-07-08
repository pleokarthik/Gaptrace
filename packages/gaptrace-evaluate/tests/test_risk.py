from gaptrace_evaluate.policy.risk import compute_risk_score
from gaptrace_evaluate.policy.schema import InputQualityPolicy


class TestRiskScore:
    def test_zero_on_clean(self):
        scores = {
            "duplicate_ratio": 0.0,
            "top_chunk_score": 0.9,
            "high_score_truncations": 0,
            "token_headroom_pct": 0.2,
            "source_domain_count": 2,
            "low_score_chunk_ratio": 0.1,
        }
        assert compute_risk_score(scores, InputQualityPolicy()) == 0.0

    def test_full_risk_all_violations(self):
        scores = {
            "duplicate_ratio": 0.5,
            "top_chunk_score": 0.3,
            "high_score_truncations": 5,
            "token_headroom_pct": 0.05,
            "source_domain_count": 10,
            "low_score_chunk_ratio": 0.8,
        }
        assert compute_risk_score(scores, InputQualityPolicy()) == 1.0

    def test_partial_risk(self):
        scores = {
            "duplicate_ratio": 0.0,
            "top_chunk_score": 0.9,
            "high_score_truncations": 5,
            "token_headroom_pct": 0.2,
            "source_domain_count": 2,
            "low_score_chunk_ratio": 0.1,
        }
        assert compute_risk_score(scores, InputQualityPolicy()) == 0.30

    def test_missing_signals_skipped(self):
        scores = {
            "duplicate_ratio": None,
            "top_chunk_score": None,
            "high_score_truncations": None,
        }
        assert compute_risk_score(scores, InputQualityPolicy()) == 0.0

    def test_empty_dict(self):
        assert compute_risk_score({}, InputQualityPolicy()) == 0.0
