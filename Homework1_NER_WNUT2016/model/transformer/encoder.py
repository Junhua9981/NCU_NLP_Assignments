import lightning.pytorch as pl
import torch.nn as nn
import torch
import math
from model.transformer.encoder_layer import EncoderLayer
from model.transformer.positional_encoding import positional_encoding

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, word_emb, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.word_emb =  torch.FloatTensor(word_emb)
    self.embedding = nn.Embedding.from_pretrained(self.word_emb)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)


    self.enc_layers = nn.ModuleList([ EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)])

    self.dropout = nn.Dropout(rate)

  def forward(self, x, mask):

    seq_len = x.shape[1]

    # adding embedding and position encoding.
    x =  self.embedding(x) # (batch_size, input_seq_len, d_model)

    x *= math.sqrt(self.d_model)
    x = (x.cpu() + self.pos_encoding[:, :seq_len, :]).to(self.device)

    x = self.dropout(x)

    # for i in range(self.num_layers):
    #   x = self.enc_layers[i](x,mask)
    for layer in self.enc_layers:
        x = layer(x, mask)

    return x  # (batch_size, input_seq_len, d_model)