from lib.search_utils import tokenize_text


def keyword_search(query: str, movies: dict) -> list:
    results = []
    query_tokens = tokenize_text(query)

    for movie in movies["movies"]:
        title_tokens = tokenize_text(movie["title"])

        if any(token in title_token for token in query_tokens for title_token in title_tokens):
            results.append(movie)

    return results