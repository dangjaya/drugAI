# import misc libs
import dill as pickle
import pandas as pd

# import from other files
from decoder import *
from   SmilesPE.pretokenizer import atomwise_tokenizer

# params
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
modelDir     = 'models/'
tmpDir       = 'temp/'
modelFile    = modelDir + 'transformerClassic.pth'
inputFile    = tmpDir + 'inputSequence.csv'
srcVocabFile = modelDir + 'SRC.pkl'
trgVocabFile = modelDir + 'TRG.pkl'
maxProtLen   = 5000
maxSMILELen  = 80

class config:
    epochs    = 25
    d_model   = 512
    n_layers  = 6
    heads     = 8
    dropout   = 0.1
    batchSize = 3
    lr        = 0.0001
    printevery= 1000

class opt:
    maxProtLen=5000
    src_pad   =1
    max_len   =80
    cut_off   =0.005

def encode(    text,
               SRC,
               TRG,
               inputFile='temp/input.csv'):
    print ("creating example of entered protein")
    df = pd.DataFrame.from_dict({'src': [text], 'trg' :['']})
    data_fields = [('src', SRC),('trg', TRG)]
    df.to_csv(inputFile, index=False,header=False)

    predict = data.TabularDataset(inputFile,format='csv', fields=data_fields )
    predict_iter = data.Iterator(predict,
                                 batch_size   = 1,
                                 device       = device)
    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']
    return predict_iter, \
           src_pad, \
           trg_pad

def addOutput(results,
              smile,
              label):
    check = validSMILES(smile)
    if check == 1:
        results.append((smile, label))
    return check


def main():
    print("Loading both protein and ligand's vocabs")
    SRC = pickle.load(open(srcVocabFile, 'rb'))
    TRG = pickle.load(open(trgVocabFile, 'rb'))

    print("Loading the trained model")
    model = torch.load(modelFile, map_location=torch.device(device))
    model.eval()

    print("Model loaded")
    protein=input("Enter protein sequence  :")
    predict_iter, src_pad, trg_pad = encode(protein, SRC, TRG)
    rangeK = 3  # top N for beam search
    results = []

    # ------------ run multiple decoding algorithms ------------ #
    for i, batch in enumerate(predict_iter):
        AAsequence = batch.src.transpose(0, 1).to(device)
        print('-' * 40 + " Greedy " + '-' * 40)
        greedySMILES = beam_greedy_search(sequence=AAsequence,
                                          model=model,
                                          smileVocab=TRG.vocab,
                                          src_pad=opt.src_pad,
                                          isGreedy=True)
        print("The resulting small molecule is  (greedy-search): %s" % greedySMILES)
        print("Valid SMILES (greedy) ", addOutput(results,
                                                  greedySMILES,
                                                  'greedy'))
        print('-' * 40 + " BEAM Search " + '-' * 40)
        for newK in range(2, rangeK):
            print("Searching Optimal SMILES with  K=%d" % newK)
            beamSMILES = beam_greedy_search(sequence=AAsequence,
                                            model=model,
                                            smileVocab=TRG.vocab,
                                            isGreedy=False,
                                            src_pad = opt.src_pad,
                                            k=newK)

            print("The resulting small molecule is  (BEAM-search) with k=%d : %s" % (newK, beamSMILES))
            print("Valid SMILES (BEAM) ", addOutput(results,
                                                    beamSMILES,
                                                    'beam k=%d' % newK))
        print('-' * 40 + " MCTS " + '-' * 40)
        mctsSMILES = MCTS(sequence=AAsequence,
                          model=model,
                          smileVocab=TRG.vocab,
                          opt=opt,
                          verbose=False)
        print("The resulting small molecule is  (MCTS-search): %s" % mctsSMILES)
        print("Valid SMILES (MCTS) ", addOutput(results,
                                                mctsSMILES,
                                                'MCTS'))


if __name__ == '__main__':
    main()

