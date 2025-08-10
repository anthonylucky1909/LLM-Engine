import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .attention import MultiheadAttn
from .block import TransformerDecoderBlock
from .LayerNormalization import LayerNormalization
from config import GPTConfig


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Token and positional embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)])

        # Final layer norm and head
        self.ln_f = LayerNormalization(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (optional)
        if getattr(cfg, "tie_word_embeddings", False):
            # embedding weight shape: (vocab_size, d_model)
            # head.weight shape: (vocab_size, d_model) because Linear(d_model, vocab_size) stores weight as (out_features, in_features)
            self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear and embedding layers
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Positional embedding typically normal init
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, T)
        attention_mask: optional mask compatible with your TransformerDecoderBlock (e.g. (B, 1, T, T) or (B, T))
        """
        batch_size, seq_len = input_ids.size()

        # Embeddings
        tok_emb = self.token_emb(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_emb[:, :seq_len, :]  # (1, T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)

        # Transformer decoder blocks
        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Autoregressive generation (simple sampling + optional top-k).

        Notes:
        - This is a minimal generator. For batching and performance, consider caching key/value states from the attention modules.
        - `temperature` must be > 0; if temperature is very small it approaches greedy sampling.
        """
        self.eval()

        device = next(self.parameters()).device
        generated = input_ids.clone().to(device)

        for _ in range(max_new_tokens):
            # Truncate if sequence grows too long
            if generated.size(1) > self.cfg.max_seq_len:
                generated = generated[:, -self.cfg.max_seq_len:]

            # Forward pass
            logits = self(generated)  # (B, T, V)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, V)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                topk_vals, topk_idx = torch.topk(next_logits, k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)  # (B, 1)
                # Set logits below threshold to -inf so they get zero prob
                next_logits = torch.where(next_logits < threshold, torch.tensor(float("-inf"), device=device), next_logits)

            # Convert to probabilities and sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping if every batch entry generated EOS
            if eos_token_id is not None:
                # next_token is (B,1); check if all are equal to eos_token_id
                if (next_token.squeeze(1) == eos_token_id).all():
                    break

        return generated
def main():
    # Step 1: Create config & model
    cfg = GPTConfig()
    model = GPTModel(cfg)

    # Step 2: Fake input (batch size 2, sequence length 5)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    print("Input IDs:\n", input_ids)

    # Step 3: Forward pass (get logits)
    logits = model(input_ids)
    print("Logits shape:", logits.shape)  # (B, T, vocab_size)

    # Step 4: Generate tokens
    generated = model.generate(input_ids, max_new_tokens=5, top_k=10)
    print("Generated IDs:\n", generated)

if __name__ == "__main__":
    main()