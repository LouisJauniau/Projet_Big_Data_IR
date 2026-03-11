"""Encode every passage in the database with ColBERT and store the
multi-vector embeddings in the ``colbert`` table.

Usage
-----
    python -m src.colbert.indexer            # default batch 64
    python -m src.colbert.indexer --batch 128
"""

import argparse
import time

import numpy as np
from psycopg2.extras import execute_values

from src.colbert.encoder import ColBERTEncoder
from src.database.connection import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers: serialise / deserialise embedding matrices for pgvector
# ---------------------------------------------------------------------------

def _tensor_to_pg_vectors(tensor) -> list[str]:
    """Convert a (n_tokens, dim) tensor to a list of pgvector-compatible strings.

    Each vector is formatted as '[v1,v2,...,vN]'.
    """
    arr = tensor.cpu().numpy()
    return [
        "[" + ",".join(f"{v:.6f}" for v in row) + "]"
        for row in arr
    ]


def _pg_array_literal(vectors: list[str]) -> str:
    """Wrap a list of pgvector strings into a PostgreSQL array literal.

    Result looks like: '{"[0.1,0.2,...]","[0.3,0.4,...]"}'
    """
    quoted = ",".join(f'"{v}"' for v in vectors)
    return "{" + quoted + "}"


# ---------------------------------------------------------------------------
# Main indexing routine
# ---------------------------------------------------------------------------

def index_passages(batch_size: int = 64):
    """Fetch passages, encode with ColBERT, and upsert into the colbert table."""

    encoder = ColBERTEncoder()

    conn = get_connection()
    cursor = conn.cursor()

    # Count passages not yet indexed
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM passages p
        WHERE NOT EXISTS (SELECT 1 FROM colbert c WHERE c.passage_id = p.id)
        """
    )
    total = cursor.fetchone()[0]
    logger.info(f"Passages to index: {total}")

    if total == 0:
        logger.info("All passages are already indexed. Nothing to do.")
        cursor.close()
        conn.close()
        return

    # Fetch passages that need indexing
    cursor.execute(
        """
        SELECT p.id, p.text
        FROM passages p
        WHERE NOT EXISTS (SELECT 1 FROM colbert c WHERE c.passage_id = p.id)
        ORDER BY p.id
        """
    )

    indexed = 0
    t0 = time.time()

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        passage_ids = [r[0] for r in rows]
        passage_texts = [r[1] for r in rows]

        # Encode each passage with ColBERT
        embeddings_list = encoder.encode_documents_batch(passage_texts)

        # Prepare rows for insertion
        insert_rows = []
        for pid, embs in zip(passage_ids, embeddings_list):
            pg_vectors = _tensor_to_pg_vectors(embs)
            pg_array = _pg_array_literal(pg_vectors)
            insert_rows.append((pid, pg_array))

        # Upsert into colbert table
        write_cur = conn.cursor()
        execute_values(
            write_cur,
            """
            INSERT INTO colbert (passage_id, embedding)
            VALUES %s
            ON CONFLICT (passage_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
            """,
            insert_rows,
            template="(%s, %s::vector[])",
            page_size=500,
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
    parser = argparse.ArgumentParser(description="Index passages with ColBERT.")
    parser.add_argument(
        "--batch", type=int, default=64,
        help="Number of passages per DB fetch batch (default: 64)",
    )
    args = parser.parse_args()
    index_passages(batch_size=args.batch)


if __name__ == "__main__":
    main()
