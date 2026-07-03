def score(hint: str, query_text: str) -> float:
    tokens = hint.lower().split()
    if not tokens:
        return 0.0
    query_lower = query_text.lower()
    matched = sum(1 for t in tokens if t in query_lower)
    return matched / len(tokens)
