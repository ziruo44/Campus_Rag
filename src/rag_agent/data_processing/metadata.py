"""Metadata extraction and enrichment utilities."""

import hashlib
import re
from typing import Optional
from langchain_core.documents import Document


def extract_college_name(header: str) -> Optional[str]:
    """从 ## 标题提取学院名，如 '## 信息工程学院' -> '信息工程学院'

    支持带序号的标题：
    - '## 信息工程学院'          -> '信息工程学院'
    - '## 6、城市学院'          -> '城市学院'
    - '## 9、通识教育学院'       -> '通识教育学院'
    """
    match = re.match(r"^##\s+(.+)$", header.strip())
    raw = match.group(1).strip() if match else None
    if not raw:
        return None
    # 去除序号前缀：阿拉伯数字+、如 6、9、
    cleaned = re.sub(r"^[\d\.]+、\s*", "", raw)
    return cleaned.strip() if cleaned else None


def extract_major_name(header: str) -> Optional[str]:
    """从 ### 标题提取专业名

    支持多种标题格式：
    - '### （1）计算机科学与技术本科专业'   -> '计算机科学与技术本科专业'
    - '### 1、工商管理'                    -> '工商管理'
    - '### 2.1.1、金融学'                   -> '金融学'
    - '### （2）数据科学与大数据技术本科专业' -> '数据科学与大数据技术本科专业'
    """
    match = re.match(r"^###\s+(.+)$", header.strip())
    raw = match.group(1).strip() if match else None
    if not raw:
        return None
    # 去除全角括号包裹的序号 （1）（2） 等
    cleaned = re.sub(r"^[（（]\d+[））]", "", raw)
    # 去除阿拉伯数字/点号序号 + 、 前缀，如 1、2.1.1、
    cleaned = re.sub(r"^[\d\.]+、\s*", "", cleaned)
    # 去除开头的全角或半角顿号（边缘情况，如 "、商务英语专业"）
    cleaned = re.sub(r"^[、、]\s*", "", cleaned)
    return cleaned.strip() if cleaned else None


def extract_section_name(header: str) -> Optional[str]:
    """从 #### 标题提取章节名，如 '#### 培养目标' -> '培养目标'"""
    match = re.match(r"^####\s+(.+)$", header.strip())
    return match.group(1).strip() if match else None


def compute_parent_id(college: str, major: str = "") -> str:
    """计算 parent_id

    学院级: MD5(college)
    专业级: MD5(college:major)
    """
    combined = f"{college}:{major}" if major else college
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def enrich_metadata(doc: Document, doc_level: str) -> Document:
    """增强文档元数据

    Args:
        doc: 文档对象
        doc_level: "college" 或 "major"

    Returns:
        增强后的文档对象
    """
    content = doc.page_content
    lines = content.split("\n")

    # 查找各级标题
    college_name = None
    major_name = None
    section_name = None

    for line in lines:
        if not line.strip():
            continue
        if college_name is None:
            college_name = extract_college_name(line)
        if major_name is None:
            major_name = extract_major_name(line)
        if section_name is None:
            section_name = extract_section_name(line)
        if college_name and major_name and section_name:
            break

    # 设置元数据
    if doc_level == "college":
        doc.metadata.update({
            "doc_type": "parent",
            "doc_level": "college",
            "college": college_name,
            "parent_id": compute_parent_id(college_name) if college_name else None,
        })
    else:  # major
        doc.metadata.update({
            "doc_type": "parent",
            "doc_level": "major",
            "college": college_name,
            "major": major_name,
            "section": section_name,
            "parent_id": compute_parent_id(college_name, major_name) if college_name and major_name else None,
        })

    return doc


def enrich_child_metadata(doc: Document, parent_id: str, chunk_index: int, college: str, major: str, section: str) -> Document:
    """为子块增强元数据

    Args:
        doc: 文档对象
        parent_id: 父文档 ID
        chunk_index: 在父文档中的顺序
        college: 学院名（从 scan 得到）
        major: 专业名（从 scan 得到，不是 chunk metadata 里的原始 专业名称）
        section: 章节类型

    Returns:
        增强后的文档对象
    """
    doc.metadata.update({
        "doc_type": "child",
        "doc_level": "major",
        "parent_id": parent_id,
        "chunk_index": chunk_index,
        "college": college,
        "major": major,
        "section": section,
    })
    return doc


if __name__ == "__main__":
    # 测试提取函数
    print("测试 extract_college_name:")
    print(f"  '## 信息工程学院' -> {extract_college_name('## 信息工程学院')}")
    print(f"  '## 金融贸易学院' -> {extract_college_name('## 金融贸易学院')}")

    print("\n测试 extract_major_name:")
    print(f"  '#### （1）计算机科学与技术本科专业' -> {extract_major_name('#### （1）计算机科学与技术本科专业')}")
    print(f"  '#### 2.1.1、金融学' -> {extract_major_name('#### 2.1.1、金融学')}")

    print("\n测试 extract_section_name:")
    print(f"  '##### 培养目标' -> {extract_section_name('##### 培养目标')}")
    print(f"  '##### 主干课程' -> {extract_section_name('##### 主干课程')}")

    print("\n测试 compute_parent_id:")
    print(f"  '信息工程学院' -> {compute_parent_id('信息工程学院')}")
    print(f"  '信息工程学院:计算机科学与技术' -> {compute_parent_id('信息工程学院', '计算机科学与技术')}")
