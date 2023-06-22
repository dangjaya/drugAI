import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torchtext import data
from torch.utils.data import DataLoader, Dataset, random_split
from SMILESTokenize import tokenize
from BatchSMILES import MyIterator, batch_size_fn
import os
import math
import dill as pickle
import json
from typing import Optional
from rdkit import Chem
from mol2vec.features import mol2alt_sentence

proteinCaches = {}

def proteinTokenizer(protSequence):
    return [protSequence[i:i + 3] for i in range(len(protSequence) - 2)]

def protein2Indices(    protVocab   ,
                        protein     ,
                        maxPROT  = 10000 ):
    if protein not in proteinCaches:
        listAAs  = [ aaa for aaa in proteinTokenizer( protein )]
        # add padding
        listAAs  += ['<unk>' for i in range( maxPROT - len( listAAs ) ) ]
        IndexSeq = [protVocab.index(eachAminoAcid) for eachAminoAcid in listAAs]
        proteinCaches[ protein ] = IndexSeq
    else:
        IndexSeq = proteinCaches[ protein ]
    return IndexSeq

def smile2Morgan( smile,
                  maxLen = 500 ):
    try:
        sentence = [ int(i) for i  in mol2alt_sentence(Chem.MolFromSmiles(smile), 1) ]
        sentence += [0] * ( maxLen - len(sentence))
        return sentence
    except Exception as e:
        print (str(e))
        print ("Unable to convert SMILE ", smile , " into MORGAN fingerprint")

class smileDataset(Dataset):

    def __init__(self,
                 df,
                 maxSMILES=1000,
                 maxPROT  =5000):
        print ("Preparing SMILE dataset")
        # Load ProtVec to convert the amino acids to indexes of ProtVec's vocabs.
        with open("data/sequenceEncoding/ProtVec.json" , 'r') as load_f:
            encoding = json.load( load_f )
            protVocab = [k for k, v in encoding.items() if k not in ('name', 'dimension')]
        print("Prepopulate timeSteps")
        self.timeSteps   = torch.tensor( df['timestep'].values )
        print("Prepopulate protein embeddings / Prot2Vec")
        listProteins     = [np.array(i) for i in df['protein'].apply( lambda protein: protein2Indices(protVocab, protein,maxPROT ) ).to_list()]
        print("-----> Convert into torch tensors")
        self.proteinLine = torch.from_numpy( np.array( listProteins ) )
        print("Prepopulate small molecule - embeddings / Mol2Vec")
        listMolecules = [np.array(i) for i in df['src'].apply(lambda smile: smile2Morgan(smile, maxSMILES)).to_list()]
        print("-----> Convert into torch tensors")
        self.drugLine    = torch.from_numpy( np.array( listMolecules ) )

    def __len__(self):
        return len(self.timeSteps)

    def __getitem__(self, idx):
        return  self.timeSteps[ idx ], \
                self.proteinLine[ idx ], \
                self.drugLine[ idx ]


def read_data(opt):
    if opt.training is not None:
        try:
            df               = pd.read_csv(opt.training)
            df.drop_duplicates(['SMILES','targetProtein'], inplace=True)
            opt.src_data     = [ i for i in df[ opt.src_col ].to_list() ]
            opt.trg_data     = [ i for i in df[ opt.trg_col ].to_list() ]
            opt.timestep     = [ i for i in df[ opt.timestep_col ].to_list()]
            opt.protein_data = [ i.upper() for i in df[opt.protein_col].to_list()]
        except Exception as e:
            print (str(e))
            quit()
    else:
        # keep target = source
        opt.trg_data = opt.src_data



def create_fields(opt):
    t_src       = tokenize()
    t_trg       = tokenize()
    t_protein   = tokenize()
    TRG     = data.Field(lower=True, tokenize=t_trg.smileTokenizer, init_token='<sos>', eos_token='<eos>')
    SRC     = data.Field(lower=True, tokenize=t_src.smileTokenizer)
    PROT    = data.Field(tokenize=t_protein.proteinTokenizer)
    TSTEP   = data.Field(sequential=False, unk_token=None)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC     = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG     = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
            PROT    = pickle.load(open(f'{opt.load_weights}/PROT.pkl', 'rb'))
            TSTEP   = pickle.load(open(f'{opt.load_weights}/TSTEP.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
    return (TSTEP, PROT, SRC, TRG)


def create_dataset(opt, TSTEP, PROT, SRC, TRG):
    print("creating dataset and iterator... ")
    raw_data = {'timestep'  : [ line for line in opt.timestep]  ,
                'target'    : [ line for line in opt.protein_data]  ,
                'src'       : [ line for line in opt.src_data]  ,
                'trg'       : [ line for line in opt.trg_data]  }

    df = pd.DataFrame(raw_data, columns=["timestep","target","src","trg"])

    mask = (df['src'].str.count(' ') < opt.max_strlen)  & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("SMILES_transformer_temp.csv",\
                                index=False,\
                                header=False)

    data_fields = [('t',TSTEP),('target',PROT),('src', SRC),('trg', TRG)]

    train = data.TabularDataset('SMILES_transformer_temp.csv',
                                format='csv',
                                fields=data_fields )

    train_iter = data.Iterator(train,
                               batch_size   = opt.batchsize,
                               device       = opt.device,
                               shuffle      =True)
    os.remove('SMILES_transformer_temp.csv')
    if opt.load_weights is None:
        TRG.build_vocab(train)
        SRC.build_vocab(train)
        TSTEP.build_vocab(train)
        PROT.build_vocab(train)
        if opt.checkpoint > 0:
            pickle.dump(SRC, open('weights_SMILES/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights_SMILES/TRG.pkl', 'wb'))
            pickle.dump(PROT, open('weights_SMILES/PROTEIN.pkl', 'wb'))
            pickle.dump(TSTEP, open('weights_SMILES/TSTEP.pkl', 'wb'))
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']
    opt.train_len = len(train_iter)
    return train_iter

# def get_len(train):
#     for i, b in enumerate(train):
#         pass
#     return i


class drugDataModule(pl.LightningDataModule):
    """
    DataModule used for training Diffusion Model for 'generating' de-novo drugs
    """

    def __init__(self, opt):
        super(drugDataModule).__init__()
        self.opt        = opt
        self.prepare_data_per_node  = False
        self._log_hyperparams       = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        # TSTEP, PROT, SRC, TRG = create_fields(self.opt)
        # batchData = create_dataset(self.opt, TSTEP, PROT, SRC, TRG)
        raw_data = {'timestep'  : [line for line in self.opt.timestep],
                    'protein'   : [line for line in self.opt.protein_data],
                    'src'       : [line for line in self.opt.src_data],
                    'trg'       : [line for line in self.opt.trg_data]}

        df = pd.DataFrame(raw_data, columns=["timestep", "protein", "src", "trg"])
        mask = (df['src'].str.count(' ') < self.opt.max_strlen) & (df['trg'].str.count(' ') < self.opt.max_strlen)
        df = df.loc[mask]

        allDataset = smileDataset( df )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            train_test_size = math.ceil( df.shape[0] * 0.7)
            test_set_size = df.shape[0] - train_test_size
            self.trainTest, self.test = random_split( allDataset, [train_test_size, test_set_size])

        if stage == "fit" or stage is None:
            train_set_size = math.ceil(len(self.trainTest) * 0.9)
            valid_set_size = len(self.trainTest) - train_set_size
            self.train, self.validate = random_split( self.trainTest, [train_set_size, valid_set_size])


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)
        #return self.train

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)
        #return self.validate

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=8)
        #return self.test

