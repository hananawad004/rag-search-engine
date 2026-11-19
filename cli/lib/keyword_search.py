import os
import pickle
import math

from collections import Counter
from lib.search_utils import load_movies,tokenize_text,tokenize_term
from lib.constants import *

CACHE_DIR = "cache"
class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.docmap={}
        self.term_frequencies = {}
        self.doc_lengths={}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self,doc_id,text):
        tokens = tokenize_text(text)

        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies.setdefault(doc_id, Counter())

        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)



    def get_tf(self,doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_idf(self, term):
        total_document_count=len(self.docmap)
        term_match_doc_count=len(self.get_documents(term))
        return math.log((total_document_count+1)/(term_match_doc_count+1))

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        N=len(self.docmap)
        df=len(self.get_documents(term))
        return math.log((N-df+.5)/(df+.5)+1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):

        tf = self.get_tf(doc_id, term)
        if tf == 0.0:
            return 0.0
        avgdl = self.__get_avg_doc_length()
        if not avgdl:
            return 0.0
        dl=self.doc_lengths.get(doc_id,0)
        length_normalization = 1 - b + b * (dl / avgdl)
        bm25_tf=(tf * (k1 + 1)) / (tf + k1 *length_normalization)
        return bm25_tf

    def build(self):
        movies = load_movies()
        self.doc_lengths.clear()
        for movie in movies:
            doc_id = movie["id"]

            self.docmap[doc_id] = movie

            self.__add_document(
                doc_id,
                f"{movie['title']} {movie['description']}"
            )

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)
    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)
def build_command():
        inverted_index = InvertedIndex()
        inverted_index.build()
        inverted_index.save()

        # docs = inverted_index.get_documents("merida")
        #
        # print(f"First document for token 'merida' = {docs[0]}")
def bm25_idf_command(term: str) -> float:
        inverted_index = InvertedIndex()
        inverted_index.load()

        token = tokenize_term(term)

        return inverted_index.get_bm25_idf(token)

def bm25_tf_command(doc_id:int,term: str,k1:float=BM25_K1,b:float=BM25_B) -> float:
        inverted_index = InvertedIndex()
        inverted_index.load()
        token=tokenize_term(term)
        return inverted_index.get_bm25_tf(doc_id,token,k1,b)

def keyword_search(query: str, inverted_index: InvertedIndex) -> list:
    results = []
    seen_doc_ids = set()

    query_tokens = tokenize_text(query)

    for token in query_tokens:
        doc_ids = inverted_index.get_documents(token)

        for doc_id in doc_ids:
            if doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(doc_id)
            results.append(inverted_index.docmap[doc_id])

            if len(results) == 5:
                return results

    return results
