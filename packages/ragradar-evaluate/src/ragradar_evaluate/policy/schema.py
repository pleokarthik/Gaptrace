from dataclasses import asdict, dataclass, fields


@dataclass
class InputQualityPolicy:
    """The thresholds ``evaluate()``/``check()`` score input quality against.

    Most callers never construct one of these — ``evaluate()`` and
    ``check()`` load the calling pipeline's stored policy automatically
    (falling back to ``InputQualityPolicy.default()`` when none is
    configured) and manage it via ``ragradar-evaluate policy set/show/
    reset``. Pass an explicit instance only to score against a one-off
    or in-memory policy instead of the persisted one. Every field is a
    threshold: values worse than it count as a policy violation.
    """

    min_chunk_relevance_score: float = 0.5
    min_top_chunk_score: float = 0.7
    max_duplicate_ratio: float = 0.2
    max_low_score_chunk_ratio: float = 0.3
    min_token_headroom: float = 0.15
    max_high_score_truncations: int = 0
    max_source_domains: int = 3
    llm_rewrite_risk_threshold: float = 0.7
    cache_borderline_margin: float = 0.03
    cache_max_age_seconds: int = 86400
    max_filtered_exclusion_ratio: float = 0.3
    min_score_variance: float = 0.0001
    min_top_second_margin: float = 0.05

    def to_dict(self) -> dict:
        """This policy as a plain, JSON-serializable dict. Pure."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InputQualityPolicy":
        """Build a policy from a dict, ignoring any unknown keys. Pure."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def default(cls) -> "InputQualityPolicy":
        """The out-of-the-box policy applied before any pipeline
        configures its own via ``ragradar-evaluate policy set``. Pure."""
        return cls()
