import argparse
from smileLightingModels import *
from ProcessSMILES import *
from Optim import CosineWithRestarts
from BatchSMILES import create_masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-training', required=True)
    parser.add_argument('-src_col', required=True)
    parser.add_argument('-trg_col', required=False)
    parser.add_argument('-timestep_col', required=False)
    parser.add_argument('-protein_col', required=False)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=100)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    opt = parser.parse_args()
    opt.device = 0 if opt.no_cuda is False else -1

    read_data(opt)
    drug_data_module = drugDataModule(opt)
    drug_data_module.setup()
    # train
    model = TransformerClassifier( opt, #len( SRC.vocab), len( TRG.vocab), # PROT.vocab,
                                   opt.d_model,
                                   opt.n_layers,
                                   opt.heads,
                                   opt.dropout )



    trainer = pl.Trainer(max_epochs=10) #, accelerator="mps")

    trainer.fit(model, drug_data_module)

    #if opt.SGDR == True:
    #    opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    #if opt.checkpoint > 0:
    #    print(
    #        "model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))
    #train_model(model, opt)


if __name__ == "__main__":
    main()
