import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PositionalEmbedding


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
        hidden_states = self.activation(self.w_1(x))
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)

        return hidden_states


class BehaviorSpecificPFF(nn.Module):
    """
    Behavior specific pointwise feedforward network.
    """

    def __init__(self, d_model, d_ff, dropout, n_b=None, bpff=False):
        super().__init__()
        self.n_b = n_b
        self.bpff = bpff
        if bpff and n_b > 1:
            self.pff = nn.ModuleList([PositionwiseFeedForward(
                d_model=d_model, d_ff=d_ff, dropout=dropout) for i in range(n_b)])
        else:
            self.pff = PositionwiseFeedForward(
                d_model=d_model, d_ff=d_ff, dropout=dropout)

    def multi_behavior_pff(self, x, b_seq):
        """
        x: B x T x H
        b_seq: B x T, 0 means padding.
        """
        outputs = [torch.zeros_like(x)]
        for i in range(self.n_b):
            outputs.append(self.pff[i](x))
        return torch.einsum('nBTh, BTn -> BTh', torch.stack(outputs, dim=0), F.one_hot(b_seq, num_classes=self.n_b+1).float())

    def forward(self, x, b_seq=None):
        if self.bpff and self.n_b > 1:
            output = self.multi_behavior_pff(x, b_seq)
        else:
            output = self.pff(x)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.LayerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor,  attention_mask):
        attention_output, _ = self.multi_head_attention(
            query=input_tensor, key=input_tensor, value=input_tensor,
            key_padding_mask=attention_mask,  # ignore padded places with True
            need_weights=False
        )
        attention_output = self.dropout(attention_output)
        return self.LayerNorm(input_tensor + attention_output)


class TransformerLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, n_b=None, bpff=False):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, d_model, dropout)
        self.feed_forward = BehaviorSpecificPFF(
            d_model=d_model, d_ff=d_ff, n_b=n_b, bpff=bpff, dropout=dropout)

    def forward(self, hidden_states, attention_mask, b_seq=None):
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output, b_seq)
        return feedforward_output


class BehaviorTransformer(nn.Module):
    def __init__(self, his_len, emb_size, num_heads, num_layers, dropout, n_b=None, bpff=False) -> None:
        super().__init__()
        self.pos_embedding = PositionalEmbedding(his_len, emb_size)
        layer = TransformerLayer(n_heads=num_heads, d_model=emb_size,
                                 d_ff=emb_size, dropout=dropout, n_b=n_b, bpff=bpff)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)])

        self.bpff = bpff

    def forward(self, his_emb: torch.Tensor, his_mask: torch.Tensor, b_seq: torch.Tensor = None):
        if self.bpff:
            b_seq = b_seq.masked_fill(his_mask, 0)

        his_encoded = his_emb + self.pos_embedding(his_emb)
        for layer_module in self.layer:
            his_encoded = layer_module(his_encoded, his_mask, b_seq)

        return his_encoded