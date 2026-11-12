import os
import pickle
import math

from collections import Counter
from lib.search_utils import load_movies,tokenize_text

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap={}
        self.term_frequencies = {}


    def __add_document(self,doc_id,text):
        tokens = tokenize_text(text)

        self.term_frequencies.setdefault(doc_id, Counter())

        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))


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

    def build(self):
        movies = load_movies()

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

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)

def build_command():
        inverted_index = InvertedIndex()
        inverted_index.build()
        inverted_index.save()

        # docs = inverted_index.get_documents("merida")
        #
        # print(f"First document for token 'merida' = {docs[0]}")


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
