import time
import numpy as np
from psycopg2.extras import execute_values
from src.database.connection import get_connection
from src.dpr.encode import get_model


def search(query, top_k=5, conn=None, model=None, log_search=True):
    """Return the top-k DPR results for *query* as a list of dictionaries."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    model = model or get_model()

    # Encode the text query into a dense vector compatible with pgvector.
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()

    cursor = conn.cursor()

    # Measure SQL retrieval latency only.
    t0 = time.time()
    cursor.execute(
        """
        SELECT d.passage_id,
               p.text,
               1 - (d.embedding <=> %s::vector) AS score
        FROM dpr d
        JOIN passages p ON p.id = d.passage_id
        ORDER BY d.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )
    fetched = cursor.fetchall()
    latency_ms = (time.time() - t0) * 1000

    results = [
        {"passage_id": pid, "text": text, "score": round(float(score), 4)}
        for pid, text, score in fetched
    ]

    if log_search:
        cursor.execute(
            """
            INSERT INTO search_logs (timestamp, algorithm, query, latency_ms)
            VALUES (NOW(), 'DPR', %s, %s)
            RETURNING id
            """,
            (query, latency_ms),
        )
        row = cursor.fetchone()
        log_id = row[0] if row is not None else 0

        execute_values(
            cursor,
            "INSERT INTO results (search_log_id, algorithm, passage_id, rank, score) VALUES %s",
            [(log_id, 'DPR', r["passage_id"], i + 1, float(r["score"])) for i, r in enumerate(results)],
        )
        conn.commit()

    cursor.close()

    if own_conn:
        conn.close()

    return results


def search_query(query, top_k=5):
    results = search(query, top_k=top_k, log_search=True)

    print(f"Query: '{query}'")
    print(f"--- TOP {top_k} RESULTS ---")
    for i, result in enumerate(results, 1):
        print(f"\n#{i} (passage_id={result['passage_id']}, score={result['score']:.4f})")
        print(result['text'])

    print("\nSearch logged")
    return results

def evaluate_mrr(eval_queries=100, eval_top_k=10):
    model = get_model()
    conn = get_connection()
    cursor = conn.cursor()

    # Evaluate only queries that have at least one relevant passage and an indexed DPR vector.
    cursor.execute(
        """
        SELECT DISTINCT q.id, q.text
        FROM queries q
        JOIN qrels qr ON qr.query_id = q.id AND qr.relevance = 1
        JOIN dpr d ON d.passage_id = qr.passage_id
        LIMIT %s
        """,
        (eval_queries,)
    )
    eval_queries_data = cursor.fetchall()
    print(f"\nQueries for evaluation: {len(eval_queries_data)}")

    if len(eval_queries_data) == 0:
        print("No query found for evaluation.")
        cursor.close()
        conn.close()
        return

    # Build a relevance lookup: query_id -> set of relevant passage_ids.
    query_ids = [q[0] for q in eval_queries_data]
    cursor.execute(
        """
        SELECT query_id, passage_id
        FROM qrels
        WHERE query_id = ANY(%s) AND relevance = 1
        """,
        (query_ids,)
    )
    relevant = {}
    for qid, pid in cursor.fetchall():
        relevant.setdefault(qid, set()).add(pid)

    # Compute reciprocal rank for each query, then average.
    rr_scores = []
    for qid, qtext in eval_queries_data:
        qemb = model.encode([qtext], convert_to_numpy=True)[0].tolist()
        cursor.execute(
            """
            SELECT d.passage_id
            FROM dpr d
            ORDER BY d.embedding <=> %s::vector
            LIMIT %s
            """,
            (qemb, eval_top_k)
        )
        retrieved = [r[0] for r in cursor.fetchall()]
        rr = 0.0
        for rank, pid in enumerate(retrieved, 1):
            if pid in relevant.get(qid, set()):
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    cursor.close()
    conn.close()

    mrr = np.mean(rr_scores)
    print(f"MRR@{eval_top_k} on {len(eval_queries_data)} queries: {mrr:.4f}")

if __name__ == "__main__":

    print("--- SEARCH TEST ---")
    search_query("Where is Paris?", top_k=5)

    print("\n--- MRR@10 EVALUATION ---")
    evaluate_mrr(eval_queries=100, eval_top_k=10)