import time
import numpy as np
from psycopg2.extras import execute_values
from src.database.connection import get_connection
from src.dpr.encode import get_model

def search_query(query, top_k=5):
    model = get_model()
    
    
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()

    conn = get_connection()
    cursor = conn.cursor()

    
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
        (query_embedding, query_embedding, top_k)
    )
    results = cursor.fetchall()
    latency_ms = (time.time() - t0) * 1000

    print(f"Requête : '{query}'")
    print(f"Latence : {latency_ms:.1f} ms\n")
    print(f"--- TOP {top_k} RÉSULTATS ---")
    for i, (pid, text, score) in enumerate(results, 1):
        print(f"\n#{i} (passage_id={pid}, score={score:.4f})")
        print(text)

    
    cursor.execute(
        """
        INSERT INTO search_logs (timestamp, algorithm, query, latency_ms)
        VALUES (NOW(), 'DPR', %s, %s)
        RETURNING id
        """,
        (query, latency_ms)
    )
    row = cursor.fetchone()
    log_id = row[0] if row is not None else 0

    execute_values(
        cursor,
        "INSERT INTO results (search_log_id, algorithm, passage_id, rank, score) VALUES %s",
        [(log_id, 'DPR', r[0], i + 1, float(r[2])) for i, r in enumerate(results)]
    )
    conn.commit()
    cursor.close()
    conn.close()

    print(f"\nRecherche loggée (search_log_id={log_id})")

def evaluate_mrr(eval_queries=100, eval_top_k=10):
    model = get_model()
    conn = get_connection()
    cursor = conn.cursor()

    
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
    print(f"\nQueries pour évaluation : {len(eval_queries_data)}")

    if len(eval_queries_data) == 0:
        print("Aucune query trouvée pour l'évaluation.")
        cursor.close()
        conn.close()
        return

    
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
    print(f"MRR@{eval_top_k} sur {len(eval_queries_data)} queries : {mrr:.4f}")

if __name__ == "__main__":
    
    print("--- TEST DE RECHERCHE ---")
    search_query("Where is Paris?", top_k=5)
    
    print("\n--- ÉVALUATION MRR@10 ---")
    evaluate_mrr(eval_queries=100, eval_top_k=10)