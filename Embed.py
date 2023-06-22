import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import json
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


# use Mol2Vec to encode SMILES into dense vector representations
class smileEmbedder(nn.Module):
    def __init__(self,
                 d_model,
                 non_trainable=True):
        super().__init__()
        try:
            # pre-trained Word2Vec embedding
            self.encoding = Word2Vec.load('data/model_300dim.pkl')
            weights = torch.FloatTensor(self.encoding.wv.vectors)
            self.embed = nn.Embedding.from_pretrained( weights )
            if non_trainable:
                self.embed.weight.requires_grad = False
            self.Linear = nn.Linear(weights.shape[1] ,d_model)
        except Exception as e:
            raise ("Error in reading protein encoding file")
            print (str(e))

    def convertIndices(self, x):
            return self.encoding.wv.key_to_index[str(x)] if x > 0 else self.encoding.wv.key_to_index['2975126068']

    def forward( self, morganSMILES ):
        # convert MORGAN fingerprints into embedding indices
        convert2Indices = np.vectorize(self.convertIndices)
        newMorganSMILES = torch.LongTensor( convert2Indices(morganSMILES.numpy()) )
        smileEmbedding  = self.embed( newMorganSMILES )
        out             = self.Linear( smileEmbedding )
        return out


class proteinEmbedder(nn.Module):

    def __init__(self,  d_model,
                        encoding_type='ProtVec',
                        non_trainable=True,
                        device='mps'):
        super().__init__()
        self.device = device
        try:
            # pre-trained embedding
            if encoding_type is not None:
                with open("data/sequenceEncoding/%s.json" % encoding_type, 'r') as load_f:
                    self.encoding   = json.load(load_f)
                    protVocab       = [k for k, v in self.encoding.items() if k not in ('name', 'dimension')]
                    protVectorSize  = len(self.encoding[protVocab[0]] )

                self.embed      = nn.Embedding( len( protVocab ), protVectorSize )
                weights_matrix  = self.construct_weight_matrix( protVocab, protVectorSize )
                self.embed.load_state_dict({'weight': torch.from_numpy(weights_matrix) })
                if non_trainable:
                    self.embed.weight.requires_grad = False
                self.embed = self.embed.to(device="mps")
            self.protLinear = nn.Linear( protVectorSize , d_model).to(device=device)
        except Exception as e:
            raise ("Error in reading protein encoding file")
            print (str(e))


    def construct_weight_matrix(self,
                                vocab,
                                vector_length=100):
        matrix_len = len(vocab)
        weights_matrix = np.zeros((matrix_len, vector_length))
        words_found = 0
        for i, word in enumerate(vocab):
            try:
                weights_matrix[i] = self.encoding[ word ] if self.encoding.__contains__( word ) else self.encoding["<unk"]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(vector_length,))
        return weights_matrix


    def forward(self,protX):
        protEmbedding = self.embed(protX)
        emb = self.protLinear( protEmbedding )
        return emb


class newPositionalEncoder( nn.Module ):

    def __init__(self,  d_model,
                        max_seq_len=10000,
                        dropout = 0.1):
        super().__init__()
        self.d_model        = d_model
        self.dropout        = nn.Dropout(dropout)
        self.max_seq_len    = max_seq_len

    def pos_encoding(self, t ):
        # pre-populate position encoding with sinusoid + cosine
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model))  # d_model
        tDupe    = np.tile( t.numpy()[:, np.newaxis, np.newaxis], (1,self.max_seq_len, self.d_model // 2) )
        #tDupe = t.repeat(1, round(self.max_seq_len / t.size(0)), self.d_model // 2)
        pos_enc_a = torch.sin(torch.from_numpy(tDupe) * inv_freq)  # bs x max_seq_len x 1/2 d_model
        pos_enc_b = torch.cos(torch.from_numpy(tDupe) * inv_freq)  # bs x max_seq_len x 1/2 d_model
        # pos_enc_a = torch.sin(t.repeat(1, round(t.shape[0] / t.size(0)), self.d_model // 2) * inv_freq)  # bs x max_seq_len x 1/2 d_model
        # pos_enc_b = torch.cos(t.repeat(1, round(t.shape[0] / t.size(0)), self.d_model // 2) * inv_freq)  # bs x max_seq_len x 1/2 d_model
        pos_enc = torch.cat([pos_enc_a, pos_enc_b],dim=2)  # concatenate the last dimension - # bs x max_seq_len x  d_model
        return pos_enc

    def forward( self,
                    x,
                    pos_enc ):
        #pos_enc = self.pos_encoding( t )

        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)         # bs x channels x d_model
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable( pos_enc[:,:seq_len], requires_grad=False)
        # pe = pe.unsqueeze(0)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 5000, dropout = 0.1):
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

#
# if overlap:
#     encodings = []
#     for i in range(len(seq) - 2):
#         encodings.append({seq[i:i + 3]: ProtVec[seq[i:i + 3]]}) if ProtVec.__contains__(
#             seq[i:i + 3]) else encodings.append({seq[i:i + 3]: ProtVec["<unk>"]})
# else:
#     encodings_1, encodings_2, encodings_3 = [], [], []
#     for i in range(0, len(seq), 3):
#         if i + 3 <= len(seq):
#             encodings_1.append({seq[i:i + 3]: ProtVec[seq[i:i + 3]]}) if ProtVec.__contains__(
#                 seq[i:i + 3]) else encodings_1.append({seq[i:i + 3]: ProtVec["<unk>"]})
#         if i + 4 <= len(seq):
#             encodings_2.append({seq[i + 1:i + 4]: ProtVec[seq[i + 1:i + 4]]}) if ProtVec.__contains__(
#                 seq[i + 1:i + 4]) else encodings_2.append({seq[i + 1:i + 4]: ProtVec["<unk>"]})
#         if i + 5 <= len(seq):
#             encodings_3.append({seq[i + 2:i + 5]: ProtVec[seq[i + 2:i + 5]]}) if ProtVec.__contains__(
#                 seq[i + 2:i + 5]) else encodings_3.append({seq[i + 2:i + 5]: ProtVec["<unk>"]})
#     encodings = [encodings_1, encodings_2, encodings_3]
# return encodings
