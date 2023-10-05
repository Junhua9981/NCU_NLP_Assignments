import torch.nn as nn

def point_wise_feed_forward_network(d_model, dff):
  return nn.Sequential(
      nn.Linear(d_model, d_model), # (batch_size, seq_len, dff)
      nn.ReLU(),
      nn.Linear(d_model, d_model)  # (batch_size, seq_len, d_model)
  )