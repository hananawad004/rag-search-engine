import argparse
from lib.semantic_search import (verify_model,embed_text,verify_embeddings,embed_query_text,SemanticSearch,
                                 semantic_search,chunk_text)
parser = argparse.ArgumentParser()

def main() -> None:

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", help="verify model")

    embed_parser = subparsers.add_parser("embed-text")
    embed_parser.add_argument("text", type=str)
    subparsers.add_parser("verify-embeddings" ,help="Build or load movie embeddings")
    query_parser = subparsers.add_parser("embed-query" ,help="Generate embedding for a search query")
    query_parser.add_argument("query", type=str)

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies semantically"
    )

    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()
    match args.command:
        case 'verify':
            verify_model()
        case 'embed-text':
            embed_text(args.text)
        case 'verify-embeddings':
            verify_embeddings()
        case 'embed-query':
            embed_query_text(args.query)
        case 'search':
            semantic_search(args.query, args.limit)

        case 'chunk':
           chunks= chunk_text(args.text, args.chunk_size)
           print(f"Chunking{len(args.text)} characters")
           for i ,chunk in enumerate(chunks,start=1):
               print(f"{i}. {chunk}")
        case _:
            parser.print_help()
if __name__ == "__main__":
    main()