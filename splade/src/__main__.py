import argparse
from src.database.connection import close_connection, init_db
from src.database.populate import populate_db


def main():
    parser = argparse.ArgumentParser(
        description="Project entry point for database initialization, population, and retrieval."
    )
    parser.add_argument(
        "command",
        choices=["init-db", "populate-db", "index-splade", "search-splade", "all"],
        nargs="?",
        default="all",
        help="Action to run. Default: all (init-db + populate-db)",
    )
    parser.add_argument("--query", type=str, help="Query text for search-splade command.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results for search.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for SPLADE indexing.")

    args = parser.parse_args()

    if args.command in ("init-db", "all"):
        conn = init_db()
        close_connection(conn)

    if args.command in ("populate-db", "all"):
        populate_db()

    if args.command == "index-splade":
        from src.splade.indexer import index_passages
        index_passages(batch_size=args.batch)

    if args.command == "search-splade":
        if not args.query:
            parser.error("--query is required for search-splade")
        from src.splade.search import search_gin
        results = search_gin(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score={r['score']:.4f}  Passage#{r['passage_id']}")
            print(f"    {r['text'][:200]}...\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
