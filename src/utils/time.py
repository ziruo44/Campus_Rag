"""Time and identifier helpers."""

from datetime import datetime, timezone
import uuid


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Create a compact random identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
