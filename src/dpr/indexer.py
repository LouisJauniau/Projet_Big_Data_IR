import time
from psycopg2.extras import execute_values
from src.database.connection import get_connection
from src.dpr.encode import get_model

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def index_passages(
    limit_passages=None,
    batch_size=128,
    db_fetch_size=2048,
    progress_every=1,
):
    """Index missing passages into the dpr table.

    Parameters
    ----------
    limit_passages : int | None
        Maximum number of missing passages to index in this run.
        Use None to index all remaining passages.
    batch_size : int
        Encoder mini-batch size.
    db_fetch_size : int
        Number of missing passages pulled from PostgreSQL per loop.
        Larger values reduce SQL overhead but use more RAM.
    progress_every : int
        Print progress every N loops.
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

    # Lazily load once; this can take time on first run when model weights are downloaded.
    model = get_model()

    started_at = time.time()
    indexed_total = 0
    loop_idx = 0

    try:
        while indexed_total < target_total:
            remaining = target_total - indexed_total
            fetch_n = min(int(db_fetch_size), int(remaining))

            cursor.execute(
                """
                SELECT p.id, p.text
                FROM passages p
                WHERE NOT EXISTS (SELECT 1 FROM dpr d WHERE d.passage_id = p.id)
                ORDER BY p.id
                LIMIT %s
                """,
                (fetch_n,),
            )
            rows = cursor.fetchall()
            if len(rows) == 0:
                break

            passage_ids = [r[0] for r in rows]
            passage_texts = [r[1] for r in rows]

            embeddings = model.encode(
                passage_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

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
                page_size=1000,
            )
            conn.commit()

            indexed_total += len(data)
            loop_idx += 1

            if loop_idx % max(1, int(progress_every)) == 0:
                elapsed = time.time() - started_at
                rate = indexed_total / elapsed if elapsed > 0 else 0.0
                eta_s = (target_total - indexed_total) / rate if rate > 0 else 0.0
                print(
                    f"[{indexed_total:,}/{target_total:,}] "
                    f"rate={rate:.1f} passages/s, "
                    f"elapsed={_format_duration(elapsed)}, "
                    f"eta={_format_duration(eta_s)}",
                    flush=True,
                )

        total_elapsed = time.time() - started_at
        print(
            f"Done! Inserted {indexed_total:,} DPR embeddings "
            f"in {_format_duration(total_elapsed)}.",
            flush=True,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Local test entry point
    index_passages(limit_passages=10_000, batch_size=128)