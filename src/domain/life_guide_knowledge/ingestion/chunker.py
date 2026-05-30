"""生活指南文档分块。"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def _clean_service_name(line: str) -> str:
    return re.sub(r"^[\d\.\s]+", "", line[3:].strip())


def _clean_sub_service_name(line: str) -> str:
    return line[4:].strip()


def chunk_life_guide_all(documents: List[Document]) -> List[Document]:
    """按服务项和子项切分生活指南。"""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "分类"),
            ("###", "服务项"),
            ("####", "子项"),
        ],
        strip_headers=False,
    )

    service_chunks: list[Document] = []
    first_category: str | None = None

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        current_category: str | None = None
        current_service_name: str | None = None

        for chunk in chunks:
            lines = chunk.page_content.split("\n")
            header_line = lines[0] if lines else ""
            header_level = len(header_line) - len(header_line.lstrip("#"))

            chunk_category: str | None = None
            chunk_service_name: str | None = None
            chunk_sub_service_name: str | None = None

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("## ") and not stripped.startswith("###") and chunk_category is None:
                    chunk_category = stripped[2:].strip()
                    continue
                if stripped.startswith("### ") and not stripped.startswith("####"):
                    chunk_service_name = _clean_service_name(stripped)
                    continue
                if stripped.startswith("#### ") and chunk_sub_service_name is None:
                    chunk_sub_service_name = _clean_sub_service_name(stripped)

            if chunk_category:
                current_category = chunk_category
                if first_category is None:
                    first_category = current_category

            if chunk_service_name:
                current_service_name = chunk_service_name

            service_category = current_category or first_category

            if chunk_service_name and header_level in (1, 2, 3):
                chunk.metadata.update(
                    {
                        "doc_type": "service",
                        "category": service_category,
                        "service_name": chunk_service_name,
                        "sub_service_name": chunk_sub_service_name,
                    }
                )
                service_chunks.append(chunk)
                continue

            if header_level == 4 and current_service_name:
                content = f"### {current_service_name}\n{chunk.page_content.strip()}"
                sub_chunk = Document(
                    page_content=content,
                    metadata=chunk.metadata.copy(),
                )
                sub_chunk.metadata.update(
                    {
                        "doc_type": "service_subitem",
                        "category": service_category,
                        "service_name": current_service_name,
                        "sub_service_name": chunk_sub_service_name,
                    }
                )
                service_chunks.append(sub_chunk)

    return service_chunks


if __name__ == "__main__":
    from .loader import load_life_guide

    docs = load_life_guide()
    chunks = chunk_life_guide_all(docs)

    print(f"服务块: {len(chunks)}")
    for i, chunk in enumerate(chunks, start=1):
        print(
            f"  {i}. [{chunk.metadata.get('category')}] "
            f"{chunk.metadata.get('service_name')} / {chunk.metadata.get('sub_service_name')}"
        )
