import torch
import torch.nn as nn
from dataclasses import dataclass
import math

@dataclass
class ModelArgs:
    vob_size = 10000
    hidden_size = 768


class TokenEmbedding(nn.Module):
    def __init__(self, vob_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vob_size,hidden_size)

    def forward(self,x):
        return self.embedding(x)



class PositionEmbedding(nn.Module):
    def __init__(self, max_seq, hidden_size):
        super().__init__()
        pe = torch.zeros(max_seq,hidden_size)

        position = torch.arange(0,max_seq)
        term = torch.exp(torch.arange(0,hidden_size,2).float()*(-math.log(10000.0)/hidden_size))

        pos = torch.einsum("i,j->ij",position,term)
        pe[:,0::2] = torch.sin(pos)
        pe[:,1::2] = torch.cos(pos)

        self.register_buffer("pe", pe)

    def forward(self,x):
        residu =x
        n = self.pe[:x.size(1),:].unsqueeze(0)
        return residu+n





batch_size = 32
seq_len = 128

# step1 embedding
x = torch.randint(0,ModelArgs.vob_size,(batch_size, seq_len))
instance = TokenEmbedding(ModelArgs.vob_size, ModelArgs.hidden_size)
b = instance(x)

# step 2
pos = PositionEmbedding(ModelArgs.vob_size, ModelArgs.hidden_size)
c = pos(b)
print(c)
