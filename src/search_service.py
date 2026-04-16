ALGORITHMS = ("splade", "colbert", "dpr")


def search_documents(algorithm, query, top_k=10, log_search=True):
    algorithm = algorithm.strip().lower()

    if algorithm == "splade":
        from splade.search import search_gin as search_splade

        return search_splade(query, top_k=top_k, log_search=log_search)
    if algorithm == "colbert":
        from colbert.search import search_bruteforce as search_colbert

        return search_colbert(query, top_k=top_k, log_search=log_search)
    if algorithm == "dpr":
        from dpr.search import search as search_dpr

        return search_dpr(query, top_k=top_k, log_search=log_search)

    raise ValueError(f"Unknown algorithm: {algorithm}")