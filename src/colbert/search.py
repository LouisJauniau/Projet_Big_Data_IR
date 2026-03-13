"""ColBERT dense retrieval: encode a query, then score all indexed passages
using the MaxSim late-interaction operator.

Two retrieval strategies are provided:

1. **Brute-force** (`search_bruteforce`):  loads all passage embeddings and
   computes MaxSim for each.  This is the natural baseline for ColBERT.

2. **Top-k** (`search_topk`):  Same as brute-force but limits to first N
   passages for quick testing.

Usage
-----
    python -m src.colbert.search "what is a bank"
    python -m src.colbert.search "how to cook pasta" --top_k 20
"""

import argparse
import time

import numpy as np
import torch

from src.colbert.encoder import ColBERTEncoder, maxsim_score, DIM
from src.database.connection import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton encoder (loaded once per process)
# ---------------------------------------------------------------------------
_encoder: ColBERTEncoder | None = None


def _get_encoder() -> ColBERTEncoder:
    global _encoder
    if _encoder is None:
        _encoder = ColBERTEncoder()
    return _encoder


# ---------------------------------------------------------------------------
# Embedding deserialisation helper
# ---------------------------------------------------------------------------

def _parse_pg_vector_array(raw) -> torch.Tensor:
    """Parse a PostgreSQL vector[] value into a (n_tokens, dim) tensor.

    psycopg2 may return the value as a string or as a list, depending on
    whether the pgvector extension is registered.  We handle both cases.
    """
    if isinstance(raw, (list, tuple)):
        # Already parsed by psycopg2 – each element is a list of floats
        # or a numpy-style string like '[0.1,0.2,...]'
        rows = []
        for item in raw:
            if isinstance(item, str):
                item = item.strip("[]")
                rows.append([float(x) for x in item.split(",")])
            elif isinstance(item, (list, tuple, np.ndarray)):
                rows.append([float(x) for x in item])
            else:
                rows.append([float(x) for x in str(item).strip("[]").split(",")])
        return torch.tensor(rows, dtype=torch.float32)

    # Fallback: raw is a string like '{"[0.1,...,0.128]","[0.2,...,0.128]"}'
    raw = str(raw)
    raw = raw.strip("{}")
    vectors_str = raw.split('","')
    rows = []
    for v in vectors_str:
        v = v.strip('"').strip("[]")
        rows.append([float(x) for x in v.split(",")])
    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Brute-force search
# ---------------------------------------------------------------------------

def search_bruteforce(
    query: str,
    top_k: int = 10,
    conn=None,
    encoder: ColBERTEncoder | None = None,
    log_search: bool = True,
) -> list[dict]:
    """Score every indexed passage against *query* using MaxSim."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    enc = encoder or _get_encoder()

    # 1. Encode query
    t0 = time.time()
    q_embs = enc.encode_query(query)
    encode_ms = (time.time() - t0) * 1000

    # 2. Fetch all indexed passage embeddings
    t1 = time.time()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.passage_id, c.embedding, p.text
        FROM colbert c
        JOIN passages p ON p.id = c.passage_id
        """
    )
    retrieval_ms = 0
    scoring_ms_total = 0

    scored = []
    for passage_id, raw_embedding, passage_text in cursor:
        t_r = time.time()
        d_embs = _parse_pg_vector_array(raw_embedding).to(enc.device)
        retrieval_ms += (time.time() - t_r) * 1000

        t_s = time.time()
        score = maxsim_score(q_embs, d_embs)
        scoring_ms_total += (time.time() - t_s) * 1000

        scored.append(
            {"passage_id": passage_id, "score": round(score, 4), "text": passage_text}
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    total_ms = (time.time() - t0) * 1000

    logger.info(
        f"ColBERT brute-force: scored {len(scored)} passages, "
        f"top-{top_k} returned (encode={encode_ms:.0f}ms, "
        f"retrieval={retrieval_ms:.0f}ms, scoring={scoring_ms_total:.0f}ms, "
        f"total={total_ms:.0f}ms)"
    )

    # 3. Log search
    if log_search:
        _log_search(conn, query, "colbert", total_ms, results)

    cursor.close()
    if own_conn:
        conn.close()

    return results


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_search(conn, query: str, algorithm: str, latency_ms: float, results: list[dict]):
    """Insert a row into search_logs and results tables."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO search_logs (timestamp, algorithm, query, latency_ms)
            VALUES (NOW(), %s, %s, %s)
            RETURNING id
            """,
            (algorithm, query, latency_ms),
        )
        log_id = cur.fetchone()[0]

        for rank, r in enumerate(results, start=1):
            cur.execute(
                """
                INSERT INTO results (search_log_id, algorithm, passage_id, rank, score)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (log_id, algorithm, r["passage_id"], rank, r["score"]),
            )
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Could not log search results: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ColBERT dense retrieval search.")
    parser.add_argument("query", type=str, help="Search query text.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results (default: 10).")
    parser.add_argument("--no-log", action="store_true", help="Don't log to search_logs table.")
    args = parser.parse_args()

    results = search_bruteforce(args.query, top_k=args.top_k, log_search=not args.no_log)

    print(f"\n{'='*80}")
    print(f"Query: {args.query}")
    print(f"Method: ColBERT (brute-force MaxSim)")
    print(f"{'='*80}\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f}  |  Passage #{r['passage_id']}")
        print(f"    {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
