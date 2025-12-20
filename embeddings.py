import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.ids = []

    def add(self, texts: list[str], ids: list[str]):
        vecs = model.encode(texts)
        self.index.add(vecs)
        self.ids.extend(ids)

    def search(self, query: str, k: int = 5):
        qvec = model.encode([query])
        D, I = self.index.search(qvec, k)
        return [self.ids[i] for i in I[0]]
