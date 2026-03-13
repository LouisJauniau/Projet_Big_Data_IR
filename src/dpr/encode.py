from sentence_transformers import SentenceTransformer


_MODEL_INSTANCE = None

def get_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        print("Chargement du modèle msmarco-distilbert-base-v4 (768 dim)...")
        _MODEL_INSTANCE = SentenceTransformer("msmarco-distilbert-base-v4")
        dimension = _MODEL_INSTANCE.get_sentence_embedding_dimension()
        print(f"Modèle chargé avec succès. Dimension : {dimension}")
    return _MODEL_INSTANCE

if __name__ == "__main__":
    
    model = get_model()