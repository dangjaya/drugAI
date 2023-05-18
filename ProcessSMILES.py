import pandas as pd
from torchtext import data
from SMILESTokenize import tokenize
from BatchSMILES import MyIterator, batch_size_fn
import os
import dill as pickle


def read_data(opt):
    if opt.training is not None:
        try:
            df           = pd.read_csv(opt.training)
            opt.src_data = [ i for i in df[ opt.src_col ].to_list() ]
            opt.trg_data = [ i for i in df[ opt.trg_col ].to_list() ]
            opt.timestep = [ i for i in df[ opt.timestep_col ].to_list()]
        except Exception as e:
            print (str(e))
            quit()
    else:
        # keep target = source
        opt.trg_data = opt.src_data


def create_fields(opt):
    t_src = tokenize()
    t_trg = tokenize()

    TRG     = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC     = data.Field(lower=True, tokenize=t_src.tokenizer)
    TSTEP   = data.Field(sequential=False, unk_token=None)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (TSTEP, SRC, TRG)


def create_dataset(opt, TSTEP, SRC, TRG):
    print("creating dataset and iterator... ")

    raw_data = {'timestep'  : [ line for line in opt.timestep]  ,
                'src'       : [ line for line in opt.src_data]  ,
                'trg'       : [ line for line in opt.trg_data]  }
    df = pd.DataFrame(raw_data, columns=["timestep","src","trg"])

    mask = (df['src'].str.count(' ') < opt.max_strlen)  & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("SMILES_transformer_temp.csv",\
                                index=False,\
                                header=False)

    data_fields = [('t',TSTEP),('src', SRC),('trg', TRG)]
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
        if opt.checkpoint > 0:
            pickle.dump(SRC, open('weights_SMILES/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights_SMILES/TRG.pkl', 'wb'))
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']
    opt.train_len = get_len(train_iter)
    return train_iter


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i
