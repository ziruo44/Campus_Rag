"""Agent 面向外层的高层工具。"""

from agent.tools.campus_navigation_tool import make_campus_navigation_tool
from agent.tools.life_guide_retrieve_tool import make_life_guide_retrieve_tool
from agent.tools.major_retrieve_tool import make_major_retrieve_tool

__all__ = [
    "make_campus_navigation_tool",
    "make_major_retrieve_tool",
    "make_life_guide_retrieve_tool",
]
