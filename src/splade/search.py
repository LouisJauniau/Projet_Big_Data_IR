"""SPLADE sparse retrieval: encode a query, then score all indexed passages
using the dot-product between sparse representations.

Two retrieval strategies are provided:

1. **SQL/GIN-based** (`search_gin`):  leverages the GIN index on
   ``splade.term_weights`` to filter candidates whose JSONB keys overlap with
   the query terms, then computes the dot-product in Python.  Good for moderate
   index sizes.

2. **Full-scan** (`search_bruteforce`):  fetches *all* sparse vectors and
   computes the dot-product in NumPy.  Simple baseline / sanity check.

Usage
-----
    python -m src.splade.search "what is a bank"
    python -m src.splade.search "how to cook pasta" --top_k 20
"""

import argparse
import json
import time

from src.database.connection import get_connection
from src.splade.encoder import SpladeEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton encoder (loaded once per process)
# ---------------------------------------------------------------------------
_encoder: SpladeEncoder | None = None


def _get_encoder() -> SpladeEncoder:
    global _encoder
    if _encoder is None:
        _encoder = SpladeEncoder()
    return _encoder


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _dot_product(query_vec: dict[str, float], passage_vec: dict[str, float]) -> float:
    """Sparse dot-product between two {token: weight} dicts."""
    score = 0.0
    # Iterate over the smaller dict for efficiency
    if len(query_vec) > len(passage_vec):
        query_vec, passage_vec = passage_vec, query_vec
    for token, q_weight in query_vec.items():
        if token in passage_vec:
            score += q_weight * passage_vec[token]
    return score


# ---------------------------------------------------------------------------
# GIN-accelerated search
# ---------------------------------------------------------------------------

def search_gin(
    query: str,
    top_k: int = 10,
    conn=None,
    encoder: SpladeEncoder | None = None,
    log_search: bool = True,
) -> list[dict]:
    """Retrieve top-k passages for *query* using GIN-filtered candidates.

    Steps:
      1. Encode query with SPLADE → sparse dict.
      2. Use ``?|`` (has-any-key) operator on the GIN index to get candidate
         passages that contain at least one query term.
      3. Compute exact dot-product for each candidate in Python.
      4. Return top-k results sorted by descending score.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    enc = encoder or _get_encoder()

    # 1. Encode query
    t0 = time.time()
    query_vec = enc.encode_single(query)
    query_terms = list(query_vec.keys())
    encode_ms = (time.time() - t0) * 1000

    if not query_terms:
        logger.warning("Query produced an empty sparse vector.")
        return []

    # 2. GIN candidate retrieval
    t1 = time.time()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.passage_id, s.term_weights, p.text
        FROM splade s
        JOIN passages p ON p.id = s.passage_id
        WHERE s.term_weights ?| %s
        """,
        (query_terms,),
    )
    candidates = cursor.fetchall()
    retrieval_ms = (time.time() - t1) * 1000

    # 3. Score candidates
    t2 = time.time()
    scored = []
    for passage_id, tw_json, passage_text in candidates:
        passage_vec = tw_json if isinstance(tw_json, dict) else json.loads(tw_json)
        score = _dot_product(query_vec, passage_vec)
        scored.append(
            {"passage_id": passage_id, "score": round(score, 4), "text": passage_text}
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    scoring_ms = (time.time() - t2) * 1000

    total_ms = encode_ms + retrieval_ms + scoring_ms
    logger.info(
        f"SPLADE search: {len(candidates)} candidates, top-{top_k} returned "
        f"(encode={encode_ms:.0f}ms, retrieval={retrieval_ms:.0f}ms, "
        f"scoring={scoring_ms:.0f}ms, total={total_ms:.0f}ms)"
    )

    # 4. Optionally log in search_logs table
    if log_search:
        _log_search(conn, query, "splade", total_ms, results)

    cursor.close()
    if own_conn:
        conn.close()

    return results


# ---------------------------------------------------------------------------
# Brute-force search (baseline)
# ---------------------------------------------------------------------------

def search_bruteforce(
    query: str,
    top_k: int = 10,
    conn=None,
    encoder: SpladeEncoder | None = None,
    log_search: bool = True,
) -> list[dict]:
    """Score every indexed passage against *query* (full scan)."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    enc = encoder or _get_encoder()

    t0 = time.time()
    query_vec = enc.encode_single(query)

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.passage_id, s.term_weights, p.text
        FROM splade s
        JOIN passages p ON p.id = s.passage_id
        """
    )

    scored = []
    for passage_id, tw_json, passage_text in cursor:
        passage_vec = tw_json if isinstance(tw_json, dict) else json.loads(tw_json)
        score = _dot_product(query_vec, passage_vec)
        scored.append(
            {"passage_id": passage_id, "score": round(score, 4), "text": passage_text}
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    total_ms = (time.time() - t0) * 1000

    logger.info(
        f"SPLADE brute-force: scored {len(scored)} passages, "
        f"top-{top_k} returned ({total_ms:.0f}ms total)"
    )

    if log_search:
        _log_search(conn, query, "splade", total_ms, results)

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
    parser = argparse.ArgumentParser(description="SPLADE sparse retrieval search.")
    parser.add_argument("query", type=str, help="Search query text.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results (default: 10).")
    parser.add_argument(
        "--method",
        choices=["gin", "bruteforce"],
        default="gin",
        help="Retrieval method (default: gin).",
    )
    parser.add_argument("--no-log", action="store_true", help="Don't log to search_logs table.")
    args = parser.parse_args()

    search_fn = search_gin if args.method == "gin" else search_bruteforce
    results = search_fn(args.query, top_k=args.top_k, log_search=not args.no_log)

    print(f"\n{'='*80}")
    print(f"Query: {args.query}")
    print(f"Method: SPLADE ({args.method})")
    print(f"{'='*80}\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f}  |  Passage #{r['passage_id']}")
        print(f"    {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
