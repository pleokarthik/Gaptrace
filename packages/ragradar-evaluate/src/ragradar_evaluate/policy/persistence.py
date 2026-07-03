from ragradar_core import store

from ragradar_evaluate.policy.schema import InputQualityPolicy


def load_policy(pipeline: str) -> InputQualityPolicy:
    """Load the stored policy for ``pipeline``, falling back to defaults.

    Read-only store access (though connecting may create/migrate the
    store). Never raises for a missing policy — returns
    ``InputQualityPolicy.default()``.
    """
    data = store.get_policy(pipeline)
    if data is None:
        return InputQualityPolicy.default()
    return InputQualityPolicy.from_dict(data)


def save_policy(pipeline: str, policy: InputQualityPolicy) -> None:
    """Persist ``policy`` for ``pipeline``. Writes to store."""
    store.write_policy(pipeline, policy.to_dict())


def reset_policy(pipeline: str) -> None:
    """Delete the stored policy for ``pipeline`` so defaults apply again.

    Writes to store. No-op if no policy was stored.
    """
    store.delete_policy(pipeline)
