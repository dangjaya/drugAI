import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, proteinEmbedder, PositionalEncoder, newPositionalEncoder
from Sublayers import Norm
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class smileProteinEncoder(nn.Module):

    def __init__(self,  vocab_size,
                        prot_vocab,
                        d_model,
                        N,
                        heads,
                        dropout):
        super().__init__()
        self.N = N
        # if the guidance is passed as a param
        if prot_vocab is not None:
            self.prot_emb = proteinEmbedder( prot_vocab, d_model )
        self.embed  = Embedder(vocab_size, d_model)
        self.pe     = newPositionalEncoder( d_model, dropout=dropout )
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm   = Norm(d_model)

    def forward(self,
                src,
                mask,
                t,
                protTarget):
        #t = t.unsqueeze(-1).type(torch.float)
        if protTarget is not None:
            pEmb = self.prot_emb( protTarget )  # bs x protein seq length x prot vector length
            t = self.pe.pos_encoding( t )[:,:pEmb.size(1),:]
            t += pEmb
        else:
            t = self.pe.pos_encoding(t)
        x = self.embed(src)
        x = self.pe(x,t)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class smileEncoder(nn.Module):

    def __init__(self,  vocab_size,
                        d_model,
                        N,
                        heads,
                        dropout):
        super().__init__()
        self.N = N
        self.embed  = Embedder(vocab_size, d_model)
        self.pe     = newPositionalEncoder( d_model, dropout=dropout )
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm   = Norm(d_model)


    def forward(self,
                src,  # bs x length
                mask,
                t):
        #t = t.unsqueeze(-1).type(torch.float)  # bs
        t = self.pe.pos_encoding(t)             # explode the dim to bs x max length x vector size
        x = self.embed(src)                     # dimension : bs x length x vector size
        x = self.pe(x,t)                        # summation : bs x length x vector size
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class smileDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = newPositionalEncoder(d_model, dropout=dropout)
        #self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask, t):
        #t = t.unsqueeze(-1).type(torch.float)
        t = self.pe.pos_encoding(t)
        x = self.embed(trg)
        x = self.pe(x,t)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed  = Embedder(vocab_size, d_model)
        self.pe     = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm   = Norm(d_model)

    def forward(self,
                src,
                mask):
        x = self.embed(src)
        x = self.pe(x)
        # x  = self.pos_encoding( x , self.time_dim )
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab,  d_model, N, heads, dropout):
        super().__init__()
        self.encoder  = smileProteinEncoder( src_vocab, d_model, N, heads, dropout )
        self.decoder = smileDecoder(trg_vocab, d_model, N, heads, dropout)
        #self.decoder  = smileProteinEncoder( trg_vocab, d_model, N, heads, dropout)
        self.encoder  = smileEncoder( src_vocab, d_model, N, heads, dropout )
        #self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        #self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self,
                src,
                trg,
                src_mask,
                trg_mask,
                t): #,protTarget):
        #e_outputs = self.encoder(src, src_mask)
        e_outputs = self.encoder(src, src_mask , t) #, protTarget)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, t)
        output = self.out(d_output)
        return output


class ProteinTransformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, prot_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder  = smileProteinEncoder( src_vocab, prot_vocab, d_model, N, heads, dropout )
        self.decoder  = smileDecoder(trg_vocab, d_model, N, heads, dropout)
        #self.decoder  = smileProteinEncoder( trg_vocab, d_model, N, heads, dropout)
        #self.encoder  = smileEncoder( src_vocab, d_model, N, heads, dropout )
        #self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        #self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out      = nn.Linear(d_model, trg_vocab)

    def forward(self,
                src,
                trg,
                src_mask,
                trg_mask,
                t,
                protTarget):
        e_outputs = self.encoder(src, src_mask , t, protTarget)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, t)
        output = self.out(d_output)
        return output


def get_model(opt,  src_vocab,
                    trg_vocab,
                    prot_vocab=None):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1
    if prot_vocab is None:
        model = Transformer(src_vocab,
                            trg_vocab,
                            opt.d_model,
                            opt.n_layers,
                            opt.heads,
                            opt.dropout)
    else:
        model = ProteinTransformer( src_vocab,
                                    trg_vocab,
                                    prot_vocab,
                                    opt.d_model,
                                    opt.n_layers,
                                    opt.heads,
                                    opt.dropout)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model





