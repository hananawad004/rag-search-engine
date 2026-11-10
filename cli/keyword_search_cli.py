import argparse
from lib.search_utils import load_movies
from lib.keyword_search import  keyword_search


def main() -> None:
    parser=argparse.ArgumentParser(description='Keyword Search CLI')
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
         print(f"Searching for: {args.query}")

         movies = load_movies();
         results = keyword_search(args.query, movies)[:5]

         for i, movie in enumerate(results, start=1):
             print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
#
# def main() -> None:
#     parser = argparse.ArgumentParser(description="Keyword Search CLI")
#     subparsers = parser.add_subparsers(dest="command", help="Available commands")
#
#     search_parser = subparsers.add_parser("search", help="Search movies using keywords")
#     search_parser.add_argument("query", type=str, help="Search query")
#
#     args = parser.parse_args()
#
#     match args.command:
#         case "search":
#             print(f"Searching for: {args.query}")
#             pass
#         case _:
#             parser.print_help()