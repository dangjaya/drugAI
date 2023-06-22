import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm

class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1,device='mps'):
        super().__init__()
        self.device = device
        self.norm_1 = Norm(d_model).to(device=self.device)
        self.norm_2 = Norm(d_model).to(device=self.device)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout).to(device=self.device)
        self.ff = FeedForward(d_model, dropout=dropout).to(device=self.device)
        self.dropout_1 = nn.Dropout(dropout).to(device=self.device)
        self.dropout_2 = nn.Dropout(dropout).to(device=self.device)
        
    def forward(self, x, mask=None):
        x2 = self.norm_1(x.to(device=self.device))
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1,device='mps'):
        super().__init__()
        self.device = device
        self.norm_1 = Norm(d_model).to(device=self.device)
        self.norm_2 = Norm(d_model).to(device=self.device)
        self.norm_3 = Norm(d_model).to(device=self.device)
        
        self.dropout_1 = nn.Dropout(dropout).to(device=self.device)
        self.dropout_2 = nn.Dropout(dropout).to(device=self.device)
        self.dropout_3 = nn.Dropout(dropout).to(device=self.device)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout).to(device=self.device)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout).to(device=self.device)
        self.ff = FeedForward(d_model, dropout=dropout).to(device=self.device)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x.to(device=self.device) )
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x