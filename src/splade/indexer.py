"""Encode every passage in the database with SPLADE and store the sparse
vectors in the ``splade`` table (``term_weights`` JSONB column).

Usage
-----
    python -m src.splade.indexer            # default batch 64
    python -m src.splade.indexer --batch 128
"""

import argparse
import json
import time

from psycopg2.extras import execute_values

from src.database.connection import get_connection
from src.splade.encoder import SpladeEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Main indexing routine
# ---------------------------------------------------------------------------

def index_passages(batch_size: int = 64, encoding_batch: int = 32):
    """Fetch passages, encode with SPLADE, and upsert into the splade table."""

    encoder = SpladeEncoder()

    conn = get_connection()
    cursor = conn.cursor()

    # Count total passages to process (exclude already indexed ones)
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM passages p
        WHERE NOT EXISTS (SELECT 1 FROM splade s WHERE s.passage_id = p.id)
        """
    )
    total = cursor.fetchone()[0]
    logger.info(f"Passages to index: {total}")

    if total == 0:
        logger.info("All passages are already indexed. Nothing to do.")
        cursor.close()
        conn.close()
        return

    # Process in batches using a server-side cursor for memory efficiency
    indexed = 0
    t0 = time.time()

    cursor.execute(
        """
        SELECT p.id, p.text
        FROM passages p
        WHERE NOT EXISTS (SELECT 1 FROM splade s WHERE s.passage_id = p.id)
        ORDER BY p.id
        """
    )

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        passage_ids = [r[0] for r in rows]
        passage_texts = [r[1] for r in rows]

        # Encode texts with SPLADE
        sparse_vecs = encoder.encode(passage_texts, batch_size=encoding_batch)

        # Prepare rows for insertion: (passage_id, term_weights_json)
        insert_rows = [
            (pid, json.dumps(svec))
            for pid, svec in zip(passage_ids, sparse_vecs)
        ]

        # Upsert into splade table
        write_cur = conn.cursor()
        execute_values(
            write_cur,
            """
            INSERT INTO splade (passage_id, term_weights)
            VALUES %s
            ON CONFLICT (passage_id) DO UPDATE
                SET term_weights = EXCLUDED.term_weights
            """,
            insert_rows,
            page_size=1000,
        )
        conn.commit()
        write_cur.close()

        indexed += len(rows)
        elapsed = time.time() - t0
        speed = indexed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Indexed {indexed}/{total} passages "
            f"({indexed * 100 / total:.1f}%) – {speed:.1f} passages/s"
        )

    elapsed = time.time() - t0
    logger.info(f"Indexing complete: {indexed} passages in {elapsed:.1f}s.")
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Index passages with SPLADE.")
    parser.add_argument(
        "--batch", type=int, default=64, help="Number of passages per DB fetch batch (default: 64)"
    )
    parser.add_argument(
        "--encoding-batch",
        type=int,
        default=32,
        help="Batch size for the SPLADE encoder (default: 32)",
    )
    args = parser.parse_args()
    index_passages(batch_size=args.batch, encoding_batch=args.encoding_batch)


if __name__ == "__main__":
    main()
