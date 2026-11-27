import numpy as np
import  os
import json
import re
from sentence_transformers import SentenceTransformer
from lib.types import ChunkMetadata

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(self.model.device)
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

    def search(self, query:str, limit:int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        results=[]
        for document,embedding in zip(self.documents,self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            results.append((score, document))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in results[:limit]
        ]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self,model_name:str = "all-MiniLM-L6-v2") -> None:
      super().__init__()
      self.chunk_embeddings=None
      self.chunk_metadata=None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents=documents
        self.document_map={
            doc['id']: doc for doc in documents
        }
        all_chunks=[]
        chunk_metadata : list[ChunkMetadata]=[]
        for movie_idx, doc in enumerate(self.documents):
            description = doc["description"]
            if not description or not description.strip():
                continue
            chunks=semantic_chunk_text(description,max_chunk_size=4,overlap=1)
            total_chunks=len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })
        self.chunk_embeddings=self.model.encode(all_chunks,show_progress_bar=True)
        self.chunk_metadata=chunk_metadata
        os.makedirs("cache", exist_ok=True)
        np.save('cache/chunk_embeddings.npy',self.chunk_embeddings)
        with open("cache/chunk_metadata.json","w") as f:
           json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents=documents
        self.document_map={ doc['id']: doc for doc in documents}
        embeddings_path="cache/chunk_embeddings.npy"
        metadata_path="cache/chunk_metadata.json"
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            self.chunk_embeddings = np.load(embeddings_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call load_or_create_chunk_embeddings first.")
        query_embedding = self.generate_embedding(query)
        chunk_scores=[]
        for chunk_embedding,metadata in zip(self.chunk_embeddings,self.chunk_metadata):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({
             "chunk_idx": metadata["chunk_idx"],
             "movie_idx":metadata["movie_idx"],
             "score": score,
            })
        movie_scores={}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        Results=[]
        for movie_idx, score in sorted_movie_scores[:limit]:
            doc = self.documents[movie_idx]
            Results.append({
            "id": doc["id"],
            "title": doc["title"],
            "document": doc["description"][:100],
            "score": round(score, 3)
            })
        return Results
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

def embed_query_text(query:str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query:str, limit:int=5):
    semantic_search = SemanticSearch()
    with open("data/movies.json", "r",encoding="utf-8") as f:
        documents = json.load(f)["movies"]
    semantic_search.load_or_create_embeddings(documents)
    results=semantic_search.search(query, limit)

    for i,result in enumerate(results,start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:200]}...")
        print()

def chunk_text(text:str,chunk_size:int=200,overlap:int=0):
    words=text.split()
    chunks=[]
    start=0
    while start<len(words):
        chunks.append(" ".join(words[start:start+chunk_size]))

        start+=chunk_size-overlap

    return chunks


def semantic_chunk_text(text:str,max_chunk_size:int=4,overlap:int=0):
    sentences=re.split((r"(?<=[.!?])\s+"),text.strip())
    chunks=[]
    start=0
    while start<len(sentences):
        chunks.append(" ".join(sentences[start:start+max_chunk_size]))
        start+=max_chunk_size-overlap
    return chunks

