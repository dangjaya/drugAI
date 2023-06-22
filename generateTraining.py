# import RDkit
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem

# import from supporting libs
from MolecularViz import MolecularVisualization

# import misc libs
import pandas as pd
import numpy as np
from scipy.stats import rv_discrete
from tqdm import tqdm
import networkx as nx
import argparse
import logging
from types import SimpleNamespace

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

config = SimpleNamespace(
            drugFileName = "data/preProcessed_BindingDB.csv"
            )

types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}

# bond 0 is no - edge or no bond
bonds           = {0: 0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
AtomCatType     = pd.CategoricalDtype(categories=[k for k, v in types.items()])
BondCatType     = pd.CategoricalDtype(categories=[k for k, v in bonds.items()])
moleculeNo      = 0
nodeColList     = [ k for k,_ in types.items()]
bondColList     = [ k for k,_ in bonds.items()]
reverseAtomType = { v:k for k,v in types.items() }
reverseBondType = { v:k for k,v in bonds.items() }
atom_decoder    = { 0 : 'C', 1 :'N', 2: 'O' , 3: 'F', 4 : 'B', 5 :'Br', 6: 'Cl', 7: 'I', 8:'P', 9:'S', 10:'Se',  11 : 'Si'}
trainings       = []
samplingSize    = 10 # number of samplings per molecule
maxMolecules    = 10000
timeSteps       = 150
mode            = 'COSINE'


def mol_to_nx(mol,
              verbose=False):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        if verbose:
            print(f'atom index {atom.GetIdx()} atom symbol {atom.GetSymbol()} convert {types[atom.GetSymbol()]}')
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol(),
                   atom_category_no=types[atom.GetSymbol()])
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if verbose:
            print(f"start {start} end {end} bondtype {bond.GetBondType()}")
        G.add_edge(start,
                   end,
                   weight=bonds[bond.GetBondType()],
                   bond_cat=bonds[bond.GetBondType()],
                   bond_type=bond.GetBondType())
    return G


# then we have to break up these molecules into list of atoms / vertices and list of connected edges
def convert2Graph( df ):
    V, E, molGraphs = [], [], []
    for i, mol in tqdm(df[:maxMolecules].iterrows(), total=maxMolecules):
        try:
            m       = Chem.MolFromSmiles(mol['smiles'])
            molNX   = mol_to_nx(m)

            atomArray       = []
            for idx in molNX.nodes():
                symbol= molNX.nodes[idx]['atom_symbol']
                atomArray.append( symbol )

            oneHot_X    = pd.get_dummies( pd.Series(atomArray, dtype=AtomCatType ) )
            adjacency_M = nx.adjacency_matrix( molNX, weight='bond_cat')
            V.append( oneHot_X )
            E.append( adjacency_M )
            molGraphs.append( { 'SMILES': mol['smiles'] ,
                                'V'     : oneHot_X,
                                'E'     : adjacency_M,
                                'target': mol['protein']})
            weights = nx.get_edge_attributes(molNX, 'weight')

        except Exception as e:
          print (f"Skipping molecule {i} {mol['smiles']}")
          print (str(e))
          V.append(None)
          E.append(None)
    return V, E, molGraphs


# Noise schedule
def noise_schedule( nb_timesteps  =  1000,
                    mode          =  "LINEAR",
                    s             =  8e-3):

    ts = np.linspace(0.0, 1.0, nb_timesteps, dtype=np.float64)
    if mode == "LINEAR":
        cum_alpha = np.cumprod(1.0 - ts) #, dim=0)
    elif mode == "COSINE":
        ft = np.power( np.cos((np.pi / 2) * (ts+s) / (1+s)),2)
        cum_alpha = ft / ft[0]
    alphas = ( cum_alpha[1:] / cum_alpha[:-1] )
    betas  = 1 - alphas
    return betas.squeeze()

def _get_full_transition_mat( betas,
                               t,
                               K):
    """Computes transition matrix for q(x_t|x_{t-1}).
    Contrary to the band diagonal version, this method constructs a transition
    matrix with uniform probability to all other states.
    Args:
      t: timestep. integer scalar.
      K: number of categories
    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = betas[t]
    mat    = np.full(shape=(K,K),
                   fill_value=beta_t/float(K),
                   dtype=np.float64)
    diag_indices = np.diag_indices_from(mat)
    diag_val = 1. - beta_t * (K-1.)/K
    mat[diag_indices] = diag_val
    # print (f'for beta {beta_t} the generated Q matrix is \n{mat}')
    return mat

def build_transition_matrix( betas,
                             transition_mat_type='uniform',
                             K=50 ,
                             num_timesteps=100):
    if transition_mat_type == 'uniform':
      q_one_step_mats = [_get_full_transition_mat(betas, t, K)
                         for t in range(0, num_timesteps)]
      q_onestep_mats = np.stack(q_one_step_mats, axis=0)
      assert q_onestep_mats.shape == (num_timesteps, K, K )
    # Construct transition matrices for q(x_t|x_start)
    q_mat_t = q_onestep_mats[0]
    q_mats = [q_mat_t]
    for t in range(1, num_timesteps):
      # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
      q_mat_t = np.dot(q_mat_t, q_onestep_mats[t]) #,axes=[[1], [0]])
      q_mats.append(q_mat_t)
    q_mats = np.stack(q_mats, axis=0)
    # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
    # Can be computed from self.q_mats and self.q_one_step_mats.
    # Only need transpose of q_onestep_mats for posterior computation.
    transpose_q_onestep_mats = np.transpose(q_onestep_mats,
                                            axes=(0, 2, 1))
    return q_mats, q_onestep_mats


def sample_discrete( probM , translate ):
    sampled_t = []
    catValues = [ v for v,_ in translate.items() ]
    for i in range(probM.shape[0]):
      probs   = probM[i,:]
      if probs.sum() > 0:
          # normalize
          probs_norm = tuple(p/sum(probs) for p in probs)
          distrib = rv_discrete(values=(catValues, probs_norm ) )
          catId   = distrib.rvs(size=1)[0]
          if translate is not None:
              sampled_t.append( translate[ catId ] )
          else:
              sampled_t.append( catId )
      else:
          # if there is no edge or neighbor for this node
          sampled_t.append( 0 )
    return np.array( sampled_t )


def saveOutputs(outputFile,
                trainings):
  outputDF   = pd.DataFrame.from_records( trainings )
  outputDF.to_csv( outputFile )


def main( conf ):
    # read the source of the data from the known protein-drug/ligand pairs ( from FDA-approved DBs such as from BindingDB )
    target_df = pd.read_csv( conf.drugFileName )
    # pre-compute beta ( noise schedule )
    betas=noise_schedule( nb_timesteps=timeSteps,
                          mode=mode)
    # Convert the ligands from the file above into RDkit Molecular structures.
    V, E, molGraphs = convert2Graph(target_df)

    print (f'Noise schedule (beta) ranging from {betas[:5]} all the way to {betas[95:]}')
    Q_X_bar, Q_X_t = build_transition_matrix( betas=betas,
                                              transition_mat_type='uniform',
                                              K=V[0].shape[1],
                                              num_timesteps=timeSteps-1)
    Q_E_bar, Q_E_t = build_transition_matrix( betas=betas,
                                              transition_mat_type='uniform',
                                              K=len(bonds),
                                              num_timesteps=timeSteps-1)

    v = MolecularVisualization(atom_decoder)
    # Iterate every molecule(s) read from the file
    for molNo, eachMol in tqdm(enumerate(molGraphs), total=len(molGraphs)):
        # Add noise(s) to each of molecule up to sampling size per mol
        for i in range(samplingSize):
            try:
                t = np.random.randint(1, timeSteps - 1)
                # # Random discrete sampling of Nodes of timestep t - 1
                probV_t_1       = np.dot(eachMol['V'], Q_X_bar[t-1])
                sampled_V_t_1   = sample_discrete(probV_t_1, reverseAtomType)
                X_t_1           = pd.get_dummies(pd.DataFrame(sampled_V_t_1, columns=['Atom'])['Atom']).astype(int).reindex(columns=nodeColList, fill_value=0)

                # # Random discrete sampling of Nodes of timestep t
                probV       = np.dot(eachMol['V'], Q_X_bar[t])
                sampled_V   = sample_discrete(probV, reverseAtomType)
                X_t         = pd.get_dummies(pd.DataFrame(sampled_V, columns=['Atom'])['Atom']).astype(int).reindex(columns=nodeColList, fill_value=0)

                # add noise to edges
                oneHot_E    = pd.get_dummies(pd.Series(eachMol['E'].todense().flatten(), dtype=BondCatType))

                # Random discrete sampling of Edges of timestep t
                probE_t_1       = np.dot(oneHot_E, Q_E_bar[t-1])
                sampled_E_t_1   = sample_discrete(probE_t_1, reverseBondType)
                E_t_1           = np.reshape(pd.DataFrame(sampled_E_t_1, columns=['Bond']).to_numpy(), E[molNo].todense().shape)

                # Random discrete sampling of Edges of timestep t-1
                probE       = np.dot(oneHot_E, Q_E_bar[t])
                sampled_E   = sample_discrete(probE, reverseBondType)
                E_t         = np.reshape(pd.DataFrame(sampled_E, columns=['Bond']).to_numpy(), E[molNo].todense().shape)

                # Convert the resulting noised one-hot encoded Atoms + noised adjacency matrix into SMILES of timestep t-1
                mol_t_1         = v.mol_from_graphs(X_t_1.idxmax(axis=1).values.tolist(), E_t_1)
                # Convert the resulting noised one-hot encoded Atoms + noised adjacency matrix into SMILES of timestep t
                mol_t           = v.mol_from_graphs(X_t.idxmax(axis=1).values.tolist(), E_t)
                trainings.append({'t'                   : t,
                                  'SMILES'              : eachMol['SMILES'],
                                  'priorNoisedSMILES'   : Chem.MolToSmiles(mol_t_1),
                                  'noisedSMILES'        : Chem.MolToSmiles(mol_t),
                                  'targetProtein'       : eachMol['target']
                                  })
            except Exception as e:
                print(str(e))
                print(f"timestep {t}")
        # save it to file
        saveOutputs(conf.outputFileName,
                    trainings)



def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--drugFileName', type=str,  help='name of the input file')
    parser.add_argument('--outputFileName', type=str,  help='output file name')
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

if __name__ == '__main__':
    parse_args(config)
    main(config)

