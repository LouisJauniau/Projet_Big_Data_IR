from psycopg2.extras import execute_values
from src.database.connection import get_connection
from src.dpr.encode import get_model

def index_passages(limit_passages=None, batch_size=128):
    """Index missing passages into the dpr table.

    Parameters
    ----------
    limit_passages : int | None
        Maximum number of missing passages to index in this run.
        Use None to index all remaining passages.
    batch_size : int
        Encoder batch size.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # Count passages that are still missing in dpr
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM passages p
        WHERE NOT EXISTS (SELECT 1 FROM dpr d WHERE d.passage_id = p.id)
        """,
    )
    row = cursor.fetchone()
    missing_total = row[0] if row is not None else 0
    print(f"Passages missing in dpr: {missing_total}")

    if missing_total == 0:
        print("All passages are already indexed. Nothing to do.")
        cursor.close()
        conn.close()
        return

    target_total = missing_total if limit_passages is None else min(missing_total, int(limit_passages))
    print(f"Indexing up to {target_total} missing passages...")

    cursor.execute(
        """
        SELECT p.id, p.text
        FROM passages p
        WHERE NOT EXISTS (SELECT 1 FROM dpr d WHERE d.passage_id = p.id)
        ORDER BY p.id
        LIMIT %s
        """,
        (target_total,)
    )
    rows = cursor.fetchall()
    if len(rows) == 0:
        # Defensive guard in case another process indexed rows between COUNT and SELECT
        print("No rows left to index after refresh. Nothing to do.")
        cursor.close()
        conn.close()
        return

    model = get_model()

    # Encode passages in batches to control memory usage
    print(f"Encoding {len(rows)} passages (this may take a few minutes)...")
    passage_ids = [r[0] for r in rows]
    passage_texts = [r[1] for r in rows]
    embeddings = model.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Insert vectors with UPSERT semantics to remain idempotent
    print("Inserting vectors into the database...")
    data = [(int(pid), emb.tolist()) for pid, emb in zip(passage_ids, embeddings)]

    execute_values(
        cursor,
        """
        INSERT INTO dpr (passage_id, embedding)
        VALUES %s
        ON CONFLICT (passage_id) DO NOTHING
        """,
        data,
        template="(%s, %s::vector)",
        page_size=500
    )
    conn.commit()
    print(f"Done! {len(data):,} embeddings were inserted.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Local test entry point
    index_passages(limit_passages=10_000, batch_size=128)