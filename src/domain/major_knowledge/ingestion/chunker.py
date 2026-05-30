"""Chunking strategies for document splitting."""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from .metadata import (
    extract_college_name,
    extract_major_name,
    extract_section_name,
    compute_parent_id,
    enrich_metadata,
    enrich_child_metadata,
)


def split_by_college(doc: Document) -> List[Document]:
    """提取学院 intro 部分（从 ## 学院名 到第一个 ### 专业 之前）

    Args:
        doc: 原始文档

    Returns:
        学院级父文档列表（只含 intro，不含专业内容）
    """
    content = doc.page_content
    lines = content.split("\n")

    college_docs = []
    current_college_name = None
    current_intro_lines = []
    in_college_block = False

    for line in lines:
        stripped = line.strip()

        # 检测 ## 标题（学院级别）- 开始新的学院
        if stripped.startswith("## ") and not stripped.startswith("###"):
            # 保存上一个学院的 intro
            if current_college_name and current_intro_lines:
                full_content = "\n".join(current_intro_lines).strip()
                if full_content:
                    college_doc = Document(
                        page_content=full_content,
                        metadata=doc.metadata.copy()
                    )
                    enrich_metadata(college_doc, "college")
                    college_docs.append(college_doc)

            # 开始新的学院
            current_college_name = extract_college_name(line)
            current_intro_lines = [line]  # 把 ## 标题行也加入 intro，供 enrich_metadata 提取学院名
            in_college_block = True

        # 检测 ### 标题（专业级别）- 学院 intro 结束
        elif stripped.startswith("### ") and not stripped.startswith("####"):
            if in_college_block and current_college_name and current_intro_lines:
                full_content = "\n".join(current_intro_lines).strip()
                if full_content:
                    college_doc = Document(
                        page_content=full_content,
                        metadata=doc.metadata.copy()
                    )
                    enrich_metadata(college_doc, "college")
                    college_docs.append(college_doc)

            current_college_name = None
            current_intro_lines = []
            in_college_block = False

        # 在学院 block 内，收集 intro 内容
        elif in_college_block and current_college_name is not None:
            current_intro_lines.append(line)

    # 保存最后一个学院的 intro（如果文件末尾没有 ### 标题）
    if current_college_name and current_intro_lines:
        full_content = "\n".join(current_intro_lines).strip()
        if full_content:
            college_doc = Document(
                page_content=full_content,
                metadata=doc.metadata.copy()
            )
            enrich_metadata(college_doc, "college")
            college_docs.append(college_doc)

    return college_docs


def split_by_major(doc: Document) -> Tuple[List[Document], List[Document]]:
    """按 #### 分割专业级父文档及其子块

    Args:
        doc: 原始文档

    Returns:
        (parent_docs, child_chunks)
    """
    # 使用 MarkdownHeaderTextSplitter 按 ### 分割
    headers_to_split_on = [
        ("#", "学院"),
        ("##", "二级学院"),
        ("###", "专业名称"),
        ("####", "章节"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    chunks = splitter.split_text(doc.page_content)

    # 分类：父文档（####级别）vs 子块（#####级别）
    parent_docs = []
    child_chunks = []

    current_college = None
    current_major = None
    current_major_parent_id = None
    prev_major = None  # 保存上一次正确的 major（不受当前 chunk 正文 scan 污染）
    prev_college = None  # 保存上一次 major 所属的学院
    major_chunk_buffer = []  # 同一专业下的所有 #### chunk

    for chunk in chunks:
        content = chunk.page_content
        lines = content.split("\n")

        # 在 chunk 的前几行查找 ##，确定所属学院
        # （因为 MarkdownHeaderTextSplitter 的 chunk 开头可能没有包含 ## 标题）
        for line in lines[:10]:  # 只在前10行查找
            if line.strip().startswith("## ") and not line.strip().startswith("###"):
                current_college = extract_college_name(line)
                break

        # 在整个 chunk 内容中查找最新的 major 标题
        for line in lines:
            if line.strip().startswith("### ") and not line.strip().startswith("####"):
                current_major = extract_major_name(line)
                current_major_parent_id = compute_parent_id(current_college, current_major)

        header_line = lines[0] if lines else ""
        header_level = len(header_line) - len(header_line.lstrip("#"))

        # 当前 chunk 的 major 来自 header（不受正文后续 ### 行影响）
        chunk_major = extract_major_name(header_line)

        if header_level == 3:  # ### 专业级父文档
            # 保存上一个专业（prev_major 是上一个 level 3 chunk 标题对应的专业名，
            # 不受当前 chunk 正文 scan 结果的影响）
            if prev_major and major_chunk_buffer:
                parent_content = "\n".join([c.page_content for c in major_chunk_buffer])
                parent_doc = Document(page_content=parent_content, metadata={})
                parent_doc.metadata.update({
                    "doc_type": "parent",
                    "doc_level": "major",
                    "college": prev_college,
                    "major": prev_major,
                    "parent_id": compute_parent_id(prev_college, prev_major),
                })
                parent_docs.append(parent_doc)

            # 开始新的专业
            major_chunk_buffer = [chunk]
            prev_major = chunk_major  # 更新 prev_major（供下一个 level 3 使用）
            prev_college = current_college  # 保存该专业所属的学院
            current_major = chunk_major
            current_major_parent_id = compute_parent_id(current_college, current_major)

        elif header_level == 4:  # #### 子块
            section = extract_section_name(header_line)
            enriched = enrich_child_metadata(
                chunk,
                parent_id=current_major_parent_id,
                chunk_index=len(major_chunk_buffer),
                college=current_college,
                major=current_major,
                section=section,
            )
            child_chunks.append(enriched)
            major_chunk_buffer.append(chunk)

    # flush: 保存最后一个专业（用 prev_major 而非 current_major）
    if prev_major and major_chunk_buffer:
        parent_content = "\n".join([c.page_content for c in major_chunk_buffer])
        parent_doc = Document(page_content=parent_content, metadata={})
        parent_doc.metadata.update({
            "doc_type": "parent",
            "doc_level": "major",
            "college": prev_college,
            "major": prev_major,
            "parent_id": compute_parent_id(prev_college, prev_major),
        })
        parent_docs.append(parent_doc)

    return parent_docs, child_chunks


def chunk_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    """两级分块：学院级 + 专业级父文档及其子块

    Args:
        documents: 原始文档列表

    Returns:
        (parent_docs, child_chunks) - 所有父文档和子块
    """
    all_parent_docs = []
    all_child_chunks = []

    for doc in documents:
        # 学院级父文档
        college_docs = split_by_college(doc)
        all_parent_docs.extend(college_docs)

        # 专业级父文档 + 子块
        major_parents, major_children = split_by_major(doc)
        all_parent_docs.extend(major_parents)
        all_child_chunks.extend(major_children)

    return all_parent_docs, all_child_chunks


if __name__ == "__main__":
    import json
    from .loader import load_documents
    from utils.paths import get_raw_data_dir

    docs = load_documents(get_raw_data_dir())
    print(f"Loaded {len(docs)} documents")

    parents, children = chunk_documents(docs)
    print(f"\nParent docs: {len(parents)}")
    print(f"Child chunks: {len(children)}")

    college_docs = [d for d in parents if d.metadata.get("doc_level") == "college"]
    major_docs = [d for d in parents if d.metadata.get("doc_level") == "major"]
    print(f"\nCollege parents: {len(college_docs)}")
    print(f"Major parents: {len(major_docs)}")

    # 打印前 5 个子块示例
    print("\n前 5 个子块示例:")
    for i, chunk in enumerate(children[:5]):
        print(f"  {i+1}. [{chunk.metadata.get('college')}] {chunk.metadata.get('major')} - {chunk.metadata.get('section')}")

    # 打印前 2 个父文档示例
    print("\n前 2 个学院级父文档:")
    for i, doc in enumerate([d for d in parents if d.metadata.get("doc_level") == "college"][:2]):
        print(f"  {i+1}. {doc.metadata.get('college')}")
        print(f"     内容长度: {len(doc.page_content)} chars")

    print("\n前 2 个专业级父文档:")
    for i, doc in enumerate([d for d in parents if d.metadata.get("doc_level") == "major"][:2]):
        print(f"  {i+1}. [{doc.metadata.get('college')}] {doc.metadata.get('major')}")
        print(f"     内容长度: {len(doc.page_content)} chars")

    # 导出到 JSON 文件
    output_path = get_raw_data_dir().parent / "chunks_debug.json"
    output_data = []
    for d in parents + children:
        output_data.append({
            "content": d.page_content,
            "metadata": d.metadata,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n已导出到 {output_path}")
