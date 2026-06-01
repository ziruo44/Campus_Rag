"""对话应用层异常定义。"""

from __future__ import annotations


class ThreadNotFoundError(FileNotFoundError):
    """查询不到线程时抛出。"""


class TurnNotFoundError(FileNotFoundError):
    """查询不到轮次时抛出。"""


class ChatExecutionError(RuntimeError):
    """单轮对话执行失败时抛出。"""
