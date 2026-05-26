"""Exception-formatting helpers."""


def format_exception_message(exc: BaseException) -> str:
    """Return the deepest non-empty exception message."""
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    if not messages:
        return exc.__class__.__name__
    return messages[-1]
