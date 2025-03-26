from ase.io import read
from ase import Atoms

class DimerObject(list):
    """
    Dimer object that takes a molecule, dimerizes it and stores the dimer, and the fragments as a list of atoms objects
    """
    def __init__(self, xyz, t=[0,0,3.0], **kwargs):
        """
        Initialize the DimerObject, with standard distance between the two molecules of 3 Angstroms 
        """
        
        self.dimer_disp = t
        self.frag1 = None 
        self.frag2 = None 
        self.dimer = None 

        self.atoms = read(xyz)
        self.mol_name = xyz.split(".")[0]

        # Dimerize 
        self.dimerize(t)   


    def dimerize(self, t):
        """
        Dimerize a molecule center it in the cell, and take the fragments of these two molecules
        """
        # Set the dimer displacement 
        self.dimer_disp = t 

        atoms_2 = self.atoms.copy()
        # Translate the molecule by the vector t
        atoms_2.translate(t)

        # Concatenate the two molecules
        atoms_dimer = self.atoms + atoms_2
        
        # Set the cell size
        atoms_dimer.set_cell([10,10,10])

        # Center the atoms in the cell
        atoms_dimer.center(axis=(0, 1, 2))#, vacuum=10.0)

        # Take the fragments of the dimer
        self.frag1 = atoms_dimer[:len(self.atoms)]
        self.frag2 = atoms_dimer[len(self.atoms):]
        self.dimer = atoms_dimer

        super().__init__([self.dimer, self.frag1, self.frag2])

    def __repr__(self):
        return f"Dimer: {self.dimer}\nFragment 1: {self.frag1}\nFragment 2: {self.frag2}"




    



