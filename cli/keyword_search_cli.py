import argparse
import sys

from lib.keyword_search import (
    InvertedIndex,
    keyword_search,
    build_command,
    bm25_idf_command,
    bm25_tf_command
)

from lib.search_utils import tokenize_term
from lib.constants import BM25_K1,BM25_B

def load_index() -> InvertedIndex:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()

    except FileNotFoundError:
        print("Error: Index files not found. Please build the index first.")
        sys.exit(1)

    return inverted_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyword Search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    # TF command
    tf_parser = subparsers.add_parser(
        "tf",
        help="Get term frequency for a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    # IDF command
    idf_parser = subparsers.add_parser(
        "idf",
        help="Calculate inverse document frequency"
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate IDF for"
    )

    # TF-IDF command
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Calculate TF-IDF score"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate TF-IDF for"
    )

    # BM25 IDF command
    bm25idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Get BM25 IDF score for a given term"
    )
    bm25idf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search",
        help="Search movies using full BM25 scoring"
    )

    bm25search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=5
    )
    # Build command
    subparsers.add_parser(
        "build",
        help="Build the inverted index"
    )

    args = parser.parse_args()

    match args.command:

        case "search":
            inverted_index = load_index()

            print(f"Searching for: {args.query}")

            results = keyword_search(args.query, inverted_index)

            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")

        case "tf":
            inverted_index = load_index()

            term = tokenize_term(args.term)
            tf = inverted_index.get_tf(args.doc_id, term)

            print(tf)

        case "idf":
            inverted_index = load_index()

            term = tokenize_term(args.term)
            idf = inverted_index.get_idf(term)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            inverted_index = load_index()

            term = tokenize_term(args.term)
            tf_idf = inverted_index.get_tfidf(args.doc_id, term)

            print(
                f"TF-IDF score of '{args.term}' "
                f"in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)

            except FileNotFoundError:
                print("Error: Index files not found. Please build the index first.")
                sys.exit(1)

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(
                    args.doc_id,
                    args.term,
                    args.k1,
                    args.b
                )
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except FileNotFoundError:
               print("Error: Index files not found. Please build the index first.")
        case "bm25search":
            inverted_index = load_index()

            results = inverted_index.bm25_search(
                args.query,
                args.limit
            )

            for i, (doc_id, score, movie) in enumerate(results, start=1):
                print(
                    f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}"
                )

        case "build":
            build_command()
            print("Index built successfully.")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()