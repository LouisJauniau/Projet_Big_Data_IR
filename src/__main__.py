import argparse
from src.database.connection import close_connection, init_db
from src.database.populate import populate_db


def main():
    parser = argparse.ArgumentParser(
        description="Project entry point for database initialization and population."
    )
    parser.add_argument(
        "command",
        choices=["init-db", "populate-db", "all", "gui"],
        nargs="?",
        default="all",
        help="Action to run. Default: all",
    )

    args = parser.parse_args()

    if args.command in ("init-db", "all"):
        conn = init_db()
        close_connection(conn)

    if args.command in ("populate-db", "all"):
        populate_db()

    if args.command == "gui":
        from src.search_gui import launch_app

        launch_app()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())