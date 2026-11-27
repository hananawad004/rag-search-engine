import argparse
import json
from lib.semantic_search import (verify_model,embed_text,verify_embeddings,embed_query_text,SemanticSearch,
                                 ChunkedSemanticSearch
                                 ,semantic_search,chunk_text,semantic_chunk_text)
from lib.search_utils import *
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
    chunk_parser.add_argument("--overlap", type=int, default=0)

    semantic_chunk_parser = subparsers.add_parser("semantic-chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    subparsers.add_parser(
        "embed-chunks",
        help="Generate embeddings for document chunks"
    )

    search_chunked_parser = subparsers.add_parser(
        "search-chunked",
        help="Search using chunk embeddings"
    )

    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)
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
           chunks= chunk_text(args.text, args.chunk_size, args.overlap)
           print(f"Chunking {len(args.text)} characters")
           for i ,chunk in enumerate(chunks,start=1):
               print(f"{i}. {chunk}")
        case 'semantic-chunk':
            chunks= semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i ,chunk in enumerate(chunks,start=1):
                print(f"{i}. {chunk}")
        case 'embed-chunks':
            with open("data/movies.json","r") as f:
                documents = json.load(f)["movies"]
                search=ChunkedSemanticSearch()
                embeddings = search.load_or_create_chunk_embeddings(documents)
                print(f"Generated {len(embeddings)} chunked embeddings")
        case "search-chunked":
            documents = load_movies()

            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(documents)

            results = search.search_chunks(args.query, args.limit)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()
if __name__ == "__main__":
    main()