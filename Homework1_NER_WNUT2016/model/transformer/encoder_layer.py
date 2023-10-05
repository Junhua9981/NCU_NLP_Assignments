import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer.layer_norm import LayerNorm
from model.transformer.multi_head_attention import MultiHeadAttention
from model.transformer.point_wise_feed_forward_network import point_wise_feed_forward_network

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.d_model = d_model
    self.dff = dff

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = LayerNorm(d_model)
    self.layernorm2 = LayerNorm(d_model)

    self.dropout1 = nn.Dropout(rate)
    self.dropout2 = nn.Dropout(rate)

  def forward(self, x, mask):
    x = x.float()
    attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1( x + attn_output )  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(x)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2