"""Model-provider health probing."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from llm.config import ChatModelSettings
from utils.errors import format_exception_message

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelProviderHealthResult:
    """Connectivity and configuration status for the upstream chat model."""

    configured: bool
    checked: bool
    reachable: bool | None
    model: str | None
    base_url: str | None
    detail: str | None = None


class ModelProviderHealthProbe:
    """Probe chat-model configuration and live connectivity."""

    def __init__(self, settings: ChatModelSettings | None = None) -> None:
        self.settings = settings or ChatModelSettings()

    def probe(self, check_connection: bool = False) -> ModelProviderHealthResult:
        """Return provider configuration and optional connectivity status."""
        model = self.settings.normalized_model
        base_url = self.settings.normalized_base_url
        api_key = self.settings.normalized_api_key
        configured = bool(base_url and api_key)

        if not configured:
            return ModelProviderHealthResult(
                configured=False,
                checked=check_connection,
                reachable=False if check_connection else None,
                model=model,
                base_url=base_url,
                detail="Missing QWEN_BASE_URL or QWEN_API_KEY.",
            )

        if not check_connection:
            return ModelProviderHealthResult(
                configured=True,
                checked=False,
                reachable=None,
                model=model,
                base_url=base_url,
                detail=None,
            )

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=10.0,
                max_retries=0,
            )
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return ModelProviderHealthResult(
                configured=True,
                checked=True,
                reachable=True,
                model=model,
                base_url=base_url,
                detail="Model provider responded successfully.",
            )
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            logger.warning("Model provider health probe failed: %s", exc)
            return ModelProviderHealthResult(
                configured=True,
                checked=True,
                reachable=False,
                model=model,
                base_url=base_url,
                detail=format_exception_message(exc),
            )
