import faiss
import numpy as np
import pickle
from typing import List
from app.models import Document

class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Document] = []

    def add_embeddings(self, embeddings: List[List[float]], documents: List[Document]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding: List[float], top_k: int = 5):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])

        return results

    def save(self, index_path: str, docs_path: str):
        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, index_path: str, docs_path: str):
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
