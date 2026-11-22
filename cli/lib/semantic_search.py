import numpy as np
import  os
import json
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings=None
        self.documents=None
        self.document_map={}

    def generate_embedding(self, text:str):
        if not text or not text.strip():
            raise ValueError('Text cannot be empty')
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents):
        self.documents=documents
        for document in documents:
            self.document_map[document['id']] = document
        texts=[f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings=self.model.encode(texts,show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save('cache/movie_embeddings.npy',self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents=documents
        for document in documents:
            self.document_map[document['id']] = document
        if os.path.exists("cache/movie_embeddings.npy"):
           self.embeddings= np.load('cache/movie_embeddings.npy')
           if len(self.embeddings)==len(documents):
             return self.embeddings
        return self.build_embeddings(documents)
def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def verify_embeddings():
    semantic_search = SemanticSearch()

    with open("data/movies.json", "r",encoding="utf-8") as f:
        documents = json.load(f)["movies"]
        embeddings = semantic_search.load_or_create_embeddings(documents)
        print(f"Number of docs:   {len(documents)}")
        print(
         f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
         )

def embed_text(text:str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")