import math
import torch
import torch.nn.functional as F
from MCTS import *
from Models import *


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    row = k_ix // k
    col = k_ix % k
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores



def init_vars(sequence,
              model,
              smileVocab,
              k=1,
              src_pad=None,
              maxProtLen=5000,
              maxSMILELen=80,
              device='cpu'):
    # outputs of encoder remain the same
    init_tok = smileVocab.stoi['<sos>']
    src_mask = (sequence != src_pad ).to(device)
    e_output = model.encoder(sequence[:,:maxProtLen],
                             src_mask[:,:maxProtLen])
    outputs  = torch.LongTensor([[init_tok]]).to(device)
    trg_mask = nopeak_mask(1).to(device)
    out      = model.out(model.decoder(   outputs,
                                          e_output,
                                          src_mask[:,:maxProtLen],
                                          trg_mask))
    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    outputs = torch.zeros(k, maxSMILELen).long()
    outputs[:, 0]   = init_tok
    outputs[:, 1]   = ix[0]
    e_outputs       = torch.zeros(k, e_output.size(-2),e_output.size(-1))
    e_outputs[:, :] = e_output[0]
    return outputs, e_outputs, log_scores


def beam_greedy_search( sequence,
                        model,
                        smileVocab,
                        isGreedy=True,
                        k=3,
                        isVerbose=True,
                        src_pad=None,
                        maxProtLen=5000,
                        maxSMILELen=80,
                        device='cpu'):
    # Greedy search is similar to Beam Search with k=1 and nullify the previously-calculated log scores
    k = 1 if isGreedy else k
    outputs, \
    e_outputs, \
    log_scores = init_vars(sequence=sequence,
                           model=model,
                           smileVocab=smileVocab,
                           k=k,
                           src_pad =src_pad,
                           maxProtLen=maxProtLen,
                           maxSMILELen=maxSMILELen,
                           device=device)

    # Obtain the indices of end-of-sentence
    eos_tok = smileVocab.stoi['<eos>']
    src_mask = (sequence != src_pad)[:,:maxProtLen].to(device)
    ind = None
    # starting after '<sos>' or start-of-sentence which is the second char
    for i in range(2, maxSMILELen):
        trg_mask = nopeak_mask(i).to(device)
        out = model.out(model.decoder(  outputs[:,:i].to(device),
                                        e_outputs.to(device),
                                        src_mask[:maxProtLen].to(device),
                                        trg_mask.to(device) )
                        )
        out = F.softmax(out, dim=-1)
        # reset the log score if this is greedy search
        log_scores = torch.zeros( log_scores.shape ) if isGreedy else log_scores
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long) #.cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    # Convert the index to ATOM based on smileVocab
    smileLookup = {v: k for k, v in smileVocab.stoi.items() }
    selectedRow = 0 if ind is None else ind
    listEOS     = (outputs[ selectedRow ] == eos_tok).nonzero()
    length      = listEOS[0].item() if listEOS.shape[0] > 0 else maxSMILELen
    listSMILES  = [smileLookup[tok] for tok in outputs[0][1:length].data.numpy() ]
    return ''.join(  listSMILES )


def MCTS( sequence,
          model,
          smileVocab,
          opt,
          device='cpu',
          verbose=True):
    contextVector,src_mask = encoderOutput( sequence,model,
                                            opt,
                                            device)
    root = MonteCarloTreeSearchNode(  model          = model,
                                      currentAtoms   = [ smileVocab.stoi['<sos>'] ],
                                      encoderContext = contextVector,
                                      source_mask    = src_mask[:, :opt.maxProtLen],
                                      eos_tok        = smileVocab.stoi['<eos>'] ,
                                      opt            = opt,
                                      cutOff         = opt.cut_off,
                                      smileVocab     = smileVocab,
                                      nodeLogProb    = 1,
                                      device         = device,
                                      verbose=verbose )
    root.expand()
    generatedMOL   = root.simulate()
    print ("best one is : ")
    print (generatedMOL)
    generatedSMILE = index2SMILE(generatedMOL,smileVocab.stoi['<eos>'], opt.max_len,  smileVocab)
    return generatedSMILE