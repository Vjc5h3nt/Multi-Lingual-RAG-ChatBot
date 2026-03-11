from typing import List
from app.models import Document

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunked_documents = []

        for doc in documents:
            text = doc.content
            words = text.split()
            start = 0
            chunk_index = 0

            while start < len(words):
                end = start + self.chunk_size
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                metadata = doc.metadata.copy()
                metadata["chunk_index"] = chunk_index

                chunked_documents.append(
                    Document(content=chunk_text, metadata=metadata)
                )

                start = end - self.overlap
                chunk_index += 1

        return chunked_documents
