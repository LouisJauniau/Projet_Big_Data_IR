from sentence_transformers import SentenceTransformer


_MODEL_INSTANCE = None

def get_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        # Lazily load once so repeated calls reuse the same model in memory.
        print("Loading model msmarco-distilbert-base-v4 (768 dim)...")
        _MODEL_INSTANCE = SentenceTransformer("msmarco-distilbert-base-v4")
        dimension = _MODEL_INSTANCE.get_sentence_embedding_dimension()
        print(f"Model loaded successfully. Dimension: {dimension}")
    return _MODEL_INSTANCE

if __name__ == "__main__":
    # Quick smoke test for manual execution.
    model = get_model()