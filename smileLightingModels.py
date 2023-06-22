from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, smileEmbedder, proteinEmbedder, PositionalEncoder, newPositionalEncoder
from Sublayers import Norm
from BatchSMILES import create_masks
import copy
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class smileEncoder(nn.Module):
    def __init__(self,  d_model,
                        N,
                        heads,
                        dropout):
        super().__init__()
        self.N = N
        self.embed  = smileEmbedder( d_model )
        self.pe     = PositionalEncoder( d_model, dropout=dropout )
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm   = Norm(d_model)

    def forward(self, src,):
        # dimensionality bs x ( len of SMILES ) x Mol2Vec vector length
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)

class smileProteinEncoder(nn.Module):
    def __init__(self,  d_model,
                        N,
                        heads,
                        dropout,
                        device='mps'):
        super().__init__()
        self.N = N
        self.embed  = proteinEmbedder( d_model )
        self.pe     = PositionalEncoder( d_model, dropout=dropout )
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm   = Norm(d_model)
        self.device = device

    def forward(self,
                src):
        src=src.to(device=self.device)
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)


class smileDecoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed  = smileEmbedder( d_model )
        self.pe     = PositionalEncoder( d_model, dropout=dropout )
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm   = Norm(d_model)

    def forward(self, trg, e_outputs, t ):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.norm(x)



class TransformerClassifier(pl.LightningModule):

    def __init__(self, opt,  d_model, N, heads, dropout):
        super().__init__()
        self.opt        = opt
        self.encoder    = smileProteinEncoder( d_model, N, heads, dropout)  #Encoder( d_model, N, heads, dropout )
        self.decoder    = smileDecoder( d_model, N, heads, dropout)
        #self.out        = nn.Linear(d_model, trg_vocab)

    def forward(self,
                src,
                trg,  #src_mask,trg_mask,
                t):
        e_outputs   = self.encoder(src) #, t ) # src_mask, t)
        d_output    = self.decoder(trg, e_outputs, t ) #src_mask, trg_mask, t)
        #output      = self.out(d_output)
        return d_output

    def cross_entropy_loss(self, preds, ys):
        preds=preds.view(-1, preds.size(-1))
        return F.cross_entropy( preds ,
                                ys,
                                ignore_index=self.opt.trg_pad)

        return  self.timeSteps[ idx ], \
                self.proteinLine[ idx ], \
                self.drugLine[ idx ]

    def training_step(self, train_batch, batch_idx):
        t, protein, drug     = train_batch
        #src                 = src.transpose(0, 1)
        #trg                 = trg.transpose(0, 1)
        #protTarget          = train_batch.target.transpose(0, 1)
        #t                   = train_batch.t
        trg_input           = drug[:, :-1]
        #src_mask, trg_mask  = create_masks(src, trg_input, self.opt)
        preds = self.forward( protein,
                              trg_input,
                              t )
                              #src_mask,
                              #trg_mask,
                              #t)
        ys = drug[:, 1:].contiguous().view(-1)
        loss = self.cross_entropy_loss(preds, ys)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
          t, protein, drug = val_batch
          #src = src.transpose(0, 1)
          #trg = trg.transpose(0, 1)
          #protTarget = val_batch.target.transpose(0, 1)
          #t = val_batch.t
          trg_input = drug[:, :-1]
          #src_mask, trg_mask = create_masks(src, trg_input, self.opt)
          preds = self.forward( protein,
                                trg_input,
                                t )
                               #src_mask,
                               #trg_mask,
                               #t) #,protTarget)
          ys = drug[:, 1:].contiguous().view(-1)
          loss = self.cross_entropy_loss(preds, ys)
          self.log('val_loss', loss)
          return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) # torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
        return optimizer

