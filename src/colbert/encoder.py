"""ColBERT v1 encoder – contextualized late-interaction over BERT.

Reference: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT", SIGIR 2020.

The encoder produces *dense, multi-vector* representations:
  - Query  → (N_q, dim) tensor  (fixed length, padded with [MASK])
  - Document → (N_d, dim) tensor (variable length, punctuation filtered)
"""

import string

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (from the paper)
# ---------------------------------------------------------------------------
DEFAULT_BERT_MODEL = "bert-base-uncased"
DIM = 128               # Projection dimension (m in the paper)
QUERY_MAXLEN = 32        # Fixed query length N_q
DOC_MAXLEN = 180         # Maximum document length


class ColBERTEncoder(nn.Module):
    """Encode queries and documents into bags of contextualised embeddings.

    - Query:    [CLS] [Q] q₀ q₁ … qₗ [MASK]…[MASK] [SEP]  →  BERT → W → L2‑norm
    - Document: [CLS] [D] d₀ d₁ … dₙ [SEP]                →  BERT → W → filter punct → L2‑norm

    Ref: §3.1 of the paper.
    """

    def __init__(
        self,
        bert_model_name: str = DEFAULT_BERT_MODEL,
        dim: int = DIM,
        device: str | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading ColBERT encoder ('{bert_model_name}') on {self.device} ...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert.eval()

        # Linear projection W ∈ R^{m × h}, no bias (as in the paper)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)

        # Punctuation token IDs (for document filtering)
        self.punctuation_ids = set(
            self.tokenizer.convert_tokens_to_ids(list(string.punctuation))
        )

        # [MASK] token id used as the "#" padding marker in queries
        self.mask_token_id = self.tokenizer.mask_token_id

        self.to(self.device)
        logger.info("ColBERT encoder loaded successfully.")

    # ------------------------------------------------------------------
    # Single-text encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_query(self, query: str, query_maxlen: int = QUERY_MAXLEN) -> torch.Tensor:
        """Encode a single query.

        f_Q(q) = Normalize(W · BERT([Q] q₀ q₁ … qₗ # … #))

        Returns:
            Tensor of shape (query_maxlen, dim) – L2-normalised.
        """
        query_tokens = self.tokenizer.tokenize(query)

        # Number of [MASK] padding tokens (#)
        n_mask = max(0, query_maxlen - len(query_tokens) - 3)  # -3: [CLS], [Q], [SEP]

        tokens = (
            ["[CLS]", "[unused0]"]
            + query_tokens[: query_maxlen - 3]
            + ["[MASK]"] * n_mask
            + ["[SEP]"]
        )

        input_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(tokens)], device=self.device
        )
        attention_mask = torch.ones_like(input_ids)

        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        projected = self.linear(hidden)
        embeddings = torch.nn.functional.normalize(projected, p=2, dim=-1)

        return embeddings.squeeze(0)  # (seq_len, dim)

    @torch.no_grad()
    def encode_doc(self, doc: str, doc_maxlen: int = DOC_MAXLEN) -> tuple[torch.Tensor, list[str]]:
        """Encode a single document.

        f_D(d) = Filter(Normalize(W · BERT([D] d₀ d₁ … dₙ)))

        Returns:
            (embeddings, filtered_tokens)
            - embeddings: Tensor (n_filtered_tokens, dim) – L2-normalised.
            - filtered_tokens: list of kept token strings.
        """
        doc_tokens = self.tokenizer.tokenize(doc)[: doc_maxlen - 3]
        tokens = ["[CLS]", "[unused1]"] + doc_tokens + ["[SEP]"]
        input_ids_list = self.tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([input_ids_list], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        projected = self.linear(hidden).squeeze(0)  # (seq_len, dim)

        # Filter out punctuation tokens
        keep_mask = [tid not in self.punctuation_ids for tid in input_ids_list]
        filtered_indices = [i for i, keep in enumerate(keep_mask) if keep]
        filtered_tokens = [tokens[i] for i in filtered_indices]
        projected = projected[filtered_indices]

        embeddings = torch.nn.functional.normalize(projected, p=2, dim=-1)
        return embeddings, filtered_tokens

    # ------------------------------------------------------------------
    # Batch encoding (for indexing)
    # ------------------------------------------------------------------
    def encode_documents_batch(
        self, texts: list[str], doc_maxlen: int = DOC_MAXLEN
    ) -> list[torch.Tensor]:
        """Encode a list of documents, returning one tensor per document.

        Each tensor has shape (n_filtered_tokens_i, dim).
        """
        results = []
        for text in texts:
            embs, _ = self.encode_doc(text, doc_maxlen=doc_maxlen)
            results.append(embs)
        return results


# ---------------------------------------------------------------------------
# MaxSim scoring (§3.2)
# ---------------------------------------------------------------------------

def maxsim_score(q_embs: torch.Tensor, d_embs: torch.Tensor) -> float:
    """Late-interaction scoring.

    S(q, d) = Σ_{i ∈ |q̂|} max_{j ∈ |d̂|} E_{q_i} · E_{d_j}^T

    Args:
        q_embs: (|q|, dim)
        d_embs: (|d|, dim)

    Returns:
        Relevance score (float).
    """
    sim_matrix = q_embs @ d_embs.T          # (|q|, |d|)
    max_sim_per_query_token, _ = sim_matrix.max(dim=-1)  # (|q|,)
    return max_sim_per_query_token.sum().item()
