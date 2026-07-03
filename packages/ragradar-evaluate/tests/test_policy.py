from ragradar_evaluate.policy.persistence import load_policy, reset_policy, save_policy
from ragradar_evaluate.policy.schema import InputQualityPolicy


class TestPolicySchema:
    def test_from_dict_ignores_unknown_keys(self):
        p = InputQualityPolicy.from_dict({"unknown_key": 99, "min_top_chunk_score": 0.8})
        assert p.min_top_chunk_score == 0.8
        assert not hasattr(p, "unknown_key")

    def test_to_dict_round_trip(self):
        p = InputQualityPolicy(min_top_chunk_score=0.8)
        p2 = InputQualityPolicy.from_dict(p.to_dict())
        assert p.to_dict() == p2.to_dict()

    def test_default(self):
        p = InputQualityPolicy.default()
        assert p.min_top_chunk_score == 0.7
        assert p.max_duplicate_ratio == 0.2


class TestPolicyStore:
    def test_default_loads_without_db(self):
        p = load_policy("unknown_pipeline")
        assert p.min_top_chunk_score == 0.7

    def test_save_and_load(self, migrated_db):
        custom = InputQualityPolicy(min_top_chunk_score=0.8, max_duplicate_ratio=0.1)
        save_policy("test_pipe", custom)
        loaded = load_policy("test_pipe")
        assert loaded.min_top_chunk_score == 0.8
        assert loaded.max_duplicate_ratio == 0.1

    def test_reset_restores_defaults(self, migrated_db):
        custom = InputQualityPolicy(min_top_chunk_score=0.99)
        save_policy("test_pipe", custom)
        reset_policy("test_pipe")
        loaded = load_policy("test_pipe")
        assert loaded.min_top_chunk_score == 0.7
