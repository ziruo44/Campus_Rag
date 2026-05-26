"""Small text-manipulation helpers."""


def truncate_text(value: str, *, limit: int) -> str:
    """Truncate text while preserving a compact ellipsis suffix."""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return f"{value[: limit - 3]}..."
