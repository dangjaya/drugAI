# import pytorch packages

from torchtext import data
import torch, gc

# import misc libs
import time
import dill as pickle
import pandas as pd
from   SmilesPE.pretokenizer import atomwise_tokenizer
import os

# import rdkit
from rdkit import Chem

# import from other files
from Models import *



# parameters
training        = 'data/preProcessed.csv'
src_col         = 'seq_0'
trg_col         = 'can_smiles'
checkpoint      = 10 # checkpoint to save the weights every N minutes
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
tempFile        = 'temp/temp_transformers.csv'
modelFile       = 'models/transformerClassic.pth'
srcVocabFile    = 'models/SRC.pkl'
trgVocabFile    = 'models/TRG.pkl'
trainTestRatio  = 0.8
trainingFile    = 'data/trainingFiles.csv'
testFile        = 'data/testFiles.csv'


# transformer hyperparameters
class config:
    epochs    = 25
    d_model   = 512
    n_layers  = 6
    heads     = 8
    dropout   = 0.1
    batchSize = 5
    lr        = 0.0001
    printevery= 100


def validSMILES(smile):
    m = Chem.MolFromSmiles(smile,sanitize=False)
    if m is None:
       print('invalid SMILES ', smile)
       return False
    else:
      try:
        Chem.SanitizeMol(m)
      except:
        print('invalid chemistry ',smile)
        return False
    return True


def read_data(savingFiles = True):
    df = pd.read_csv( training )
    print ("Input file has %d pairs" % df.shape[0] )
    df.drop_duplicates([ trg_col, src_col ], inplace=True)
    print ("number of de-duped training pairs before sanitization ", df.shape[0])
    validMask = df[trg_col].apply( lambda smi : validSMILES(smi))
    df = df[ validMask ]
    print ("number of training pairs after sanitization ", df.shape[0])
    print ("Split-up training & test sets")
    proteinDF = df.drop_duplicates( subset=['seq_0'] )
    proteinTrainDF = proteinDF.sample( frac= trainTestRatio )
    mergeDF = df.merge( proteinTrainDF, on=['seq_0'], suffixes=('','_y') , how='left' )
    trainDF = mergeDF[ ~pd.isnull( mergeDF['can_smiles_y'] ) ]
    testDF  = mergeDF[ pd.isnull( mergeDF['can_smiles_y'] ) ]
    print ('Training set has %d pairs' % ( trainDF.shape[0] ) )
    print ('Test set has %d pairs' % ( testDF.shape[0] ) )
    if savingFiles:
        trainDF.to_csv( trainingFile )
        testDF.to_csv( testFile )
    src_data, trg_data = [], []
    for i, eachPair in trainDF.iterrows():
        src_data.append( eachPair[src_col] )
        trg_data.append( eachPair[trg_col] )
    return src_data, trg_data


class tokenize(object):
    def smileTokenizer(self, smi):
        return atomwise_tokenizer(smi)

    def proteinTokenizer(self,
                         protSequence):
        encoding_data = [ aa.upper() for aa in protSequence]
        return encoding_data


def create_fields():
    t_src       = tokenize()
    t_trg       = tokenize()
    TRG         = data.Field(tokenize=t_trg.smileTokenizer, init_token='<sos>', eos_token='<eos>')
    SRC         = data.Field(tokenize=t_src.proteinTokenizer )
    return (SRC, TRG)


def create_dataset(src_data,
                   trg_data,
                   SRC,
                   TRG):
    print("creating dataset and iterator... ")
    raw_data = {'src'       : [ line for line in src_data]  ,
                'trg'       : [ line for line in trg_data]  }
    df = pd.DataFrame(raw_data, columns=["src","trg"])
    df.to_csv(tempFile, index=False, header=False)
    data_fields = [('src', SRC),('trg', TRG)]
    train = data.TabularDataset(tempFile,
                                format='csv',
                                fields=data_fields)
    train_iter = data.Iterator(train,
                               batch_size   = config.batchSize,
                               device       = device,
                               shuffle      = True)
    os.remove(tempFile)
    TRG.build_vocab(train)
    SRC.build_vocab(train)
    if checkpoint > 0:
        pickle.dump(SRC, open(srcVocabFile, 'wb'))
        pickle.dump(TRG, open(trgVocabFile, 'wb'))
    src_pad   = SRC.vocab.stoi['<pad>']
    trg_pad   = TRG.vocab.stoi['<pad>']
    train_len = len(train_iter)
    print ("Training length ", train_len)
    return train_iter, src_pad, trg_pad, train_len


# code from AllenNLP

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs

def train_model(model,
                trainData,
                config,
                src_pad,
                trg_pad,
                SGDR=False,
                verbose=True):
    print("training model...")
    model.train()
    start = time.time()
    if checkpoint > 0:
        cptime = time.time()

    for epoch in range(config.epochs):
        # Clear un-used memory prior to training the model
        gc.collect()
        torch.cuda.empty_cache()
        # train
        print ("EPOCH - %d" % epoch)
        total_loss = 0
        if verbose:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

        if checkpoint > 0:
            torch.save(model, modelFile)

        for i, batch in enumerate(trainData):
            if verbose and (i+1) % config.printevery ==0:
               print (f"--> batch {i}")
            src                 = batch.src.transpose(0,1).to(device)
            trg                 = batch.trg.transpose(0,1).to(device)
            trg_input           = trg[:, :-1]
            src_mask, trg_mask  = create_masks(src=src,
                                               src_pad=src_pad,
                                               trg=trg_input,
                                               trg_pad=trg_pad)
            preds               = model(src,
                                        trg_input,
                                        src_mask,
                                        trg_mask)
            ys                  = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
            loss.backward()
            optimizer.step()
            if SGDR == True:
                sched.step()

            total_loss += loss.item()

            if (i + 1) % config.printevery == 0:
                 p = int(100 * (i + 1) / train_len)
                 avg_loss = total_loss/config.printevery
                 print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                 ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0

            if checkpoint > 0 and ((time.time()-cptime)//60) // checkpoint >= 1:
                # save the entire model and not just the weights
                torch.save( model,  modelFile )
                cptime = time.time()
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))
    # save the model
    torch.save( model,  modelFile )


if __name__ == '__main__':
    src_data, trg_data = read_data()
    SRC, TRG  = create_fields()
    train, SRC_PAD,TRG_PAD, train_len = create_dataset(src_data, trg_data, SRC, TRG)
    model = get_model(config, len(SRC.vocab), len(TRG.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    sched     = CosineWithRestarts(optimizer, T_max=train_len)
    train_model(model,
                train,
                config,
                SRC_PAD,
                TRG_PAD)

