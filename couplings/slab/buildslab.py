import numpy as np 
from ase.io import read, write 
import numpy as np 
import pickle 
from ase.atoms import Atoms 
import os 



class BuildSlab(list):
    def __init__(self, slab: Atoms, molecule: Atoms, slab_mol_dist: float, mol_site: int, pbc: list):
        """
        Build the slab-molecule system
        Args:
            slab: Atoms object, the slab
            molecule: Atoms object, the molecule
            slab_mol_dist: float, the distance between the slab and the molecule
            mol_site: int, the adsorption site of the molecule
            pbc: list, the periodic boundary condition
        """

        self.slab = slab
        self.molecule = molecule
        self.slab_mol_dist = slab_mol_dist
        self.mol_site = mol_site
        self.pbc = pbc

        self.frag_12 = None
        self.frag_1 = None
        self.frag_2 = None
        self.frags = ["frag12", "frag1", "frag2"]

        self._merge_slab_mol()




    def _merge_slab_mol(self):

        """
        Merge the slab and molecule, and get the fragments
        """

        mol = self.molecule.copy()
        slab = self.slab.copy()


        # 
        current_cell = slab.get_cell() 
        new_cell = np.array([current_cell[0], current_cell[1], np.array([0, 0, 15])])
        slab.set_cell(new_cell)
        slab.translate([0, 0, 3])  # move the slab to the positive z direction
        lattice_vectors = slab.get_cell()


        pos_surf_ads = slab[self.mol_site].position
        pos_ads = pos_surf_ads + np.array( [0, 0, self.slab_mol_dist] )
        mol.translate(pos_ads)  # move the molecule to the adsorption site

        mol_slab = mol + slab   
        mol_slab.set_pbc(self.pbc)
        mol_slab.set_cell(lattice_vectors)

        self.frag_1 = mol_slab[:len(mol)] # molecule
        self.frag_2 = mol_slab[len(mol):] # slab
        self.frag_12 = mol_slab.copy() # molecule + slab

        super().__init__([self.frag_12, self.frag_1, self.frag_2])
  
    



















