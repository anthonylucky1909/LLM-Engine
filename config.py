from typing import Optional

class GPTConfig:
    def __init__(
        self,
        vocab_size: int = 13025,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 5,
        dropout: float = 0.1,
        tie_word_embeddings: bool = True,
        use_bias_in_proj: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_bias_in_proj = use_bias_in_proj