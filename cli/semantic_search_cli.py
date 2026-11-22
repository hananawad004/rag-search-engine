import argparse
from lib.semantic_search import verify_model,embed_text
parser = argparse.ArgumentParser()

def main() -> None:

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", help="verify model")

    embed_parser = subparsers.add_parser("embed-text")
    embed_parser.add_argument("text", type=str)

    args = parser.parse_args()
    match args.command:
        case "-":
            parser.print_help()
        case 'verify':
            verify_model()
        case 'embed-text':
            embed_text(args.text)
if __name__ == "__main__":
    main()