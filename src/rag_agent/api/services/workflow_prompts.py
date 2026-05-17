"""Prompt helpers for the explicit workflow runtime."""

from __future__ import annotations

WORKFLOW_ANSWER_SYSTEM_PROMPT = """
你是温州商学院知识库问答助手。

你只能根据提供的检索结果和对话上下文回答用户最后一个问题，不能编造知识库中不存在的信息。
如果检索结果不足以回答，请明确说明“根据当前检索结果，暂无足够信息回答这个问题”。
不要提及工具、工作流、检索过程或模型。
回答使用中文，保持简洁清晰，不要使用 emoji。
如果是对比或区别问题，请按清晰维度组织答案。
""".strip()


def build_final_answer_user_prompt(query: str, retrieval_context: str) -> str:
    """Build the final user prompt for answer generation."""
    return (
        f"用户最后一个问题：\n{query}\n\n"
        f"检索结果：\n{retrieval_context}\n\n"
        "请直接给出最终回答。"
    )
