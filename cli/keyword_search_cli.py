
import argparse
import sys

from lib.keyword_search import InvertedIndex, keyword_search, build_command
from lib.search_utils import tokenize_term


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    tf_parser = subparsers.add_parser(
        "tf",
        help="Get term frequency for a document"
    )

    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )

    tf_parser.add_argument(
        "term",
        type=str,
        help="Term to count"
    )
    idf_parser = subparsers.add_parser(
        "idf",
        help="Calculate inverse document frequency"
    )

    idf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate IDF for"
    )
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Calculate TF-IDF score"
    )

    tfidf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )

    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate TF-IDF for"
    )
    subparsers.add_parser(
        "build",
        help="Build the inverted index"
    )


    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            inverted_index = InvertedIndex()

            try:
                inverted_index.load()

            except FileNotFoundError:
                print("Error: Index files not found. Please build the index first.")
                sys.exit(1)

            results = keyword_search(args.query, inverted_index)

            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")

        case "tf":
            inverted_index = InvertedIndex()

            try:
                inverted_index.load()

            except FileNotFoundError:
                print("Error: Index files not found. Please build the index first.")
                sys.exit(1)

            term = tokenize_term(args.term)

            tf = inverted_index.get_tf(args.doc_id, term)

            print(tf)
        case "idf":
            inverted_index = InvertedIndex()

            try:
                inverted_index.load()
            except FileNotFoundError:
                print("Error: Index files not found. Please build the index first.")
                sys.exit(1)

            term = tokenize_term(args.term)
            idf = inverted_index.get_idf(term)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            inverted_index = InvertedIndex()

            try:
                inverted_index.load()
            except FileNotFoundError:
                print("Error: Index files not found. Please build the index first.")
                sys.exit(1)

            term = tokenize_term(args.term)

            tf_idf = inverted_index.get_tfidf(args.doc_id, term)

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "build":
            build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
