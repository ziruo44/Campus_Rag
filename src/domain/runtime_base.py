"""共享运行时基类。"""

from __future__ import annotations

import logging
from threading import Lock

from shared.observability.performance import measure_stage

logger = logging.getLogger(__name__)


class LazyRuntimeBase:
    """带锁的惰性初始化运行时基类。"""

    def __init__(
        self,
        *,
        stage_prefix: str,
        failure_message: str,
        log_name: str,
    ) -> None:
        self._lock = Lock()
        self._initialized = False
        self._stage_prefix = stage_prefix
        self._failure_message = failure_message
        self._log_name = log_name

    @property
    def is_initialized(self) -> bool:
        """是否已经完成初始化。"""
        return self._initialized

    def ensure_initialized(self) -> None:
        """按需执行一次初始化。"""
        if self._initialized:
            return

        with measure_stage(f"{self._stage_prefix}.ensure_initialized"):
            with self._lock:
                if self._initialized:
                    return

                try:
                    self._initialize_once()
                    self._initialized = True
                except Exception as exc:
                    logger.exception("Failed to initialize %s", self._log_name)
                    raise self.runtime_error_class(self._failure_message) from exc

    @property
    def runtime_error_class(self):
        """返回当前运行时对应的异常类型。"""
        raise NotImplementedError

    def _initialize_once(self) -> None:
        """由子类实现实际初始化过程。"""
        raise NotImplementedError
