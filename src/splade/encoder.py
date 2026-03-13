import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default model – SPLADE v2 (CoCondenser + ensemble distillation)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"


class SpladeEncoder:
    """Encode text into SPLADE sparse representations.

    Each text is mapped to a dictionary  {token_string: weight}  where only
    tokens with weight > 0 are kept (sparse representation).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading SPLADE model '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("SPLADE model loaded successfully.")

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
        """Return a list of sparse vectors (one per input text).

        Each sparse vector is a dict mapping vocabulary tokens to their
        SPLADE weight (only non-zero entries).
        """
        all_sparse_vecs: list[dict[str, float]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**tokens).logits  # (B, seq_len, vocab_size)

            # SPLADE aggregation: max over sequence length, then log(1 + ReLU(x))
            relu_log = torch.log1p(torch.relu(logits))
            # Max-pool over token positions → (B, vocab_size)
            sparse_vecs, _ = torch.max(relu_log * tokens["attention_mask"].unsqueeze(-1), dim=1)

            # Convert each vector to a {token: weight} dict (keep non-zero only)
            for vec in sparse_vecs:
                indices = vec.nonzero(as_tuple=True)[0]
                weights = vec[indices]
                token_ids = indices.cpu().numpy().tolist()
                weight_vals = weights.cpu().numpy().tolist()

                sparse_dict: dict[str, float] = {}
                for tid, w in zip(token_ids, weight_vals):
                    token_str = self.tokenizer.decode([tid]).strip()
                    if token_str:  # skip empty tokens
                        sparse_dict[token_str] = round(float(w), 4)

                all_sparse_vecs.append(sparse_dict)

        return all_sparse_vecs

    def encode_single(self, text: str) -> dict[str, float]:
        """Convenience wrapper for a single text."""
        return self.encode([text])[0]
