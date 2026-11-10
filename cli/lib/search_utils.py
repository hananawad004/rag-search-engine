import json
import string
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def load_movies() -> dict:
    with open("data/movies.json", "r") as file:
        return json.load(file)


def tokenize(text: str) -> list:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()


def load_stopwords() -> list:
    with open("data/stopwords.txt") as file:
        return file.read().split()


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list:
    return [stemmer.stem(word) for word in tokenize(text) if word not in STOPWORDS]