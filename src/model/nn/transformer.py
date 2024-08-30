import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, config, size):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        self.emb_in = nn.Linear(config.seq_len - 1, config.hidden_ndim)
        self.pe = RotaryEmbedding(config.hidden_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_ndim, config.nheads, config.dropout, size
                )
                for _ in range(config.nlayers)
            ]
        )
        self.emb_out = nn.Linear(config.hidden_ndim, config.seq_len - 1)

    def forward(self, x):
        # x (b, seq_len, pt, b)
        b, seq_len, pt, d = x.size()

        x = x.view(b, seq_len, pt * d)
        x = x.permute(0, 2, 1)  # (b, pt*d, seq_len)
        x = self.emb_in(x)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        for layer in self.encoders:
            x, attn_w = layer(x)

        x = self.emb_out(x)
        x = x.permute(0, 2, 1)  # (b, seq_len, pt*d)
        x = x.view(b, seq_len, pt, d)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout, size):
        super().__init__()
        size = size[0] * size[1]

        self.attn = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.GroupNorm(1, size)

        self.ff = SwiGLU(ndim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(1, size)

    def forward(self, x, need_weights=False):
        # x (b, seq_len, pt * d, hidden_ndim)
        x_attn, attn_w = self.attention_block(x, need_weights)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.feed_forward_block(x))

        return x, attn_w

    def attention_block(self, x, need_weights):
        x, attn_w = self.attn(x, x, x, need_weights=need_weights)
        return x, attn_w

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout2(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_ndim: int, out_ndim: int = None):
        super().__init__()

        if out_ndim is None:
            out_ndim = in_ndim
        hdim = int(in_ndim * 4 * (2 / 3))
        self.w1 = nn.Linear(in_ndim, hdim)
        self.w2 = nn.Linear(hdim, out_ndim)
        self.w3 = nn.Linear(in_ndim, hdim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
