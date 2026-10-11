"""Chunking: breaking the long questions only."""

from langchain_core.documents import Document


def chunk_qwise(all_docs: list[dict]) -> list[Document]:
    chunks: list[Document] = []

    for item in all_docs:
        content = item["content"]

        if len(content) > 1000:
            for i in range(0, len(content), 800):
                chunk_text = content[i : i + 1000]
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={**item["metadata"], "chunk_index": i // 800},
                    )
                )
        else:
            chunks.append(Document(page_content=content, metadata=item["metadata"]))

    print(f"Total {len(all_docs)} questions -> {len(chunks)} chunks")
    return chunks
