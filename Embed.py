import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class newPositionalEncoder( nn.Module ):

    def __init__(self,  d_model,
                        max_seq_len=500,
                        dropout = 0.1):
        super().__init__()
        self.d_model        = d_model
        self.dropout        = nn.Dropout(dropout)
        self.max_seq_len    = max_seq_len

    def  forward( self, x, t ):
        # pre-populate position encoding with sinusoid + cosine
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model ))  # d_model
        pos_enc_a = torch.sin( t.repeat( 1, round(self.max_seq_len / t.size(0)), self.d_model // 2) * inv_freq )  # bs x max_seq_len x 1/2 d_model
        pos_enc_b = torch.cos( t.repeat( 1, round(self.max_seq_len / t.size(0)), self.d_model // 2) * inv_freq)    # bs x max_seq_len x 1/2 d_model
        self.pos_enc   = torch.cat( [pos_enc_a, pos_enc_b], dim=2 )  # concatenate the last dimension - # bs x max_seq_len x  d_model

        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)         # bs x channels x d_model
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable( self.pos_enc[:,:seq_len], requires_grad=False)
        # pe = pe.unsqueeze(0)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 500, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))   # check if the bracket is needed
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)