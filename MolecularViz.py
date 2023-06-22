from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
import os

class MolecularVisualization:

    def __init__(self,  atom_decoder):
        # dictionary to map integer value to the char of atom
        self.atom_decoder = atom_decoder

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # create empty editable mol object
        mol = Chem.RWMol()
        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(node_list[i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx
#            print (f"add node {i} with type {node_list[i]}")

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond==0:
                    continue
                elif bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 12:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                #                     print (f"bond {bond}")
                # print (f"Add bond from {node_to_idx[ix]} to {node_to_idx[iy]} with type of bond {bond_type}")
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, trainer=None, log='graph',toFile=True):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        images = []
        for i in range(num_molecules_to_visualize):
            try:
              file_path = os.path.join(path, 'molecule_{}.png'.format(i))
              mol = self.mol_from_graphs( molecules[i][0],  molecules[i][1] )
              if toFile:
                  Draw.MolToFile(mol, file_path)
              else:
                  images.append( Draw.MolToImage( mol ) )

            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")
        if not toFile:
          return images