import numpy as np 
import os 
import pickle as pickle 
from gpaw import GPAW
from gpaw.lcao.tools import get_lcao_hamiltonian
from collections import defaultdict
import scipy.linalg


class MolFODFT:
    def __init__(self, dimerobject: list, **kwargs):

        """
        Initialize Fragment-Orbital DFT object from a DimerObject (list)
        """

        self.dimerobject = dimerobject # This is the dimerobject
        self.dimer_distance = scipy.linalg.norm(dimerobject.dimer_disp, 2)

        self.HS_matrices = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.HS_diabats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.frags = ["dimer", "frag1", "frag2"]

        os.makedirs("fodft", exist_ok=True) # Check curr direct
        

    def calculate_HS(self, xc: str, basis: str, **kwargs): 
        """
        Calculate the Hamiltonian and Overlap matrices for the dimer and fragments
        """
        # Update directories for the current choice of xc and basis
        self._update_directories(xc, basis)

        # Loop over Atoms-objects in the dimerobject: [dimer, frag1, frag2]
        for i, frag in enumerate(self.dimerobject):
            txt_path = f"fodft/calc_dir/{self.dimerobject.mol_name}/xc_{xc}/basis_{basis}/dist_{self.dimer_distance}/{self.dimerobject.mol_name}_{self.frags[i]}.txt"

            frag.pbc = [False, False, False] # Make sure PBC is turned off

            frag.calc = GPAW( #asasda sd asd 
                xc=xc,
                mode='lcao', # LCAO mode
                basis=basis,
                kpts=[1, 1, 1],  # Use only the Gamma point: G. Kastlunger recommendation
                txt=txt_path
            )

            # Perform the calculation to get the potential energy
            frag.get_potential_energy() #asdasd a s
            # Get the LCAO Hamiltonian and overlap matrices
            H_skMM, S_kMM = get_lcao_hamiltonian(frag.calc)
            H_MM = H_skMM[0][0] # Only use the first k-point and spin (molecular)
            S_MM = S_kMM[0] # Only use the first k-point (molecular)
            self.HS_matrices[xc][basis][self.dimer_distance][self.frags[i]] = { "H": H_MM, 
                                                                                "S": S_MM, 
                                                                                "n_bands": frag.calc.get_number_of_bands(),
                                                                                "nelec": frag.calc.get_number_of_electrons(),
                                                                                "occupations": frag.calc.get_occupation_numbers(),
                                                                                "homolumo": frag.calc.get_homo_lumo()
                                                                            }

        print("Calculated HS matrices for", self.dimerobject.mol_name, "at distance", self.dimer_distance, "with xc:", xc, "and basis:", basis)

    def calculate_HS_all(self, xc_list: list, basis_list: list, **kwargs):
        """
        Calculate the Hamiltonian and Overlap matrices for the dimer and fragments for all xc and basis
        """
        for xc in xc_list:
            for basis in basis_list:
                self.calculate_HS(xc, basis)

        print("Calculated HS matrices for all xc and basis for", self.dimerobject.mol_name, "at distance", self.dimer_distance)

    def calculate_couplings_matrix(self, xc: str, basis: str):
        """
        Calculate couplings with given functional and basis
        """
        try:
            if self.HS_matrices[xc][basis][self.dimer_distance]:
                print("Calculating and building the block-diagonal Hamiltonian")
                # print(f"Calculating coupling matrix in the diabatic basis")
                # print(f"HS matrices for xc: {xc}, basis: {basis}, and distance: {self.dimer_distance} already exist.")
            else:
                raise KeyError
        except KeyError:
            print(f"HS matrices for: \n Functional: {xc} \n Basis: {basis} \n Dimer distance: {self.dimer_distance} does not exist.\n This will be done first")
            # Calculate HS matrices if they do not exist
            self.calculate_HS(xc, basis)

        
        # Load Hamiltonian and overlap matrices
        H_dimer = self.HS_matrices[xc][basis][self.dimer_distance]["dimer"]["H"]
        S_dimer = self.HS_matrices[xc][basis][self.dimer_distance]["dimer"]["S"]
        
        Hfrag1_orig = self.HS_matrices[xc][basis][self.dimer_distance]["frag1"]["H"]
        Sfrag1_orig = self.HS_matrices[xc][basis][self.dimer_distance]["frag1"]["S"]

        Hfrag2_orig = self.HS_matrices[xc][basis][self.dimer_distance]["frag2"]["H"]
        Sfrag2_orig = self.HS_matrices[xc][basis][self.dimer_distance]["frag2"]["S"]

        # Löwdin orthogonalization matrix
        Xfrag1 = scipy.linalg.sqrtm(scipy.linalg.inv(Sfrag1_orig))
        Xfrag2 = scipy.linalg.sqrtm(scipy.linalg.inv(Sfrag2_orig))

        # Eigenvalues and -vectors of fragments
        f1_eigvals_orig, f1_eigvecs_orig = scipy.linalg.eigh(Hfrag1_orig, Sfrag1_orig)
        f2_eigvals_orig, f2_eigvecs_orig = scipy.linalg.eigh(Hfrag2_orig, Sfrag2_orig)

        # Löwdin basis transformation
        Hfrag1_lowdin = Xfrag1.T @ Hfrag1_orig @ Xfrag1
        Hfrag2_lowdin = Xfrag2.T @ Hfrag1_orig @ Xfrag2 

        # Sanity check for overlap matrix
        Sfrag1_lowdin = Xfrag1.T @ Sfrag1_orig @ Xfrag1
        Sfrag2_lowdin = Xfrag2.T @ Sfrag2_orig @ Xfrag2
        identity = np.identity(Sfrag1_lowdin.shape[0])
        assert np.allclose(Sfrag1_lowdin, identity) & np.allclose(Sfrag2_lowdin, identity)

        # Diagonalize the Löwdin orthogonalized Hamiltonian matrices
        f1_eigvals_lowdin, f1_eigvecs_lowdin = scipy.linalg.eigh(Hfrag1_lowdin, Sfrag1_lowdin)
        f2_eigvals_lowdin, f2_eigvecs_lowdin = scipy.linalg.eigh(Hfrag2_lowdin, Sfrag2_lowdin)
        
        # Eigenvalues should be the same for the Löwdin orthogonalized and non-orthogonalized Hamiltonian matrices
        assert np.allclose(f1_eigvals_lowdin, f1_eigvals_orig) & np.allclose(f2_eigvals_lowdin, f2_eigvals_orig)
        
        # Fragment eigenvectors in original basis
        va_orig = f1_eigvecs_orig
        vd_orig = f2_eigvecs_orig

        # List of column vectors with the long dimension of the total dimer Hamiltonian
        VA_orig = []
        VD_orig = []
        
        # Dimension of acceptor and donor molecules
        Na = len(Hfrag1_orig)
        Nd = len(Hfrag2_orig)

        # Zero pad the eigenvectors, such that they have (Na + Nd) columns and Na and Nd rows
        for iac in range(Na):
            va = va_orig[:, iac] 
            va = np.concatenate( (va, np.zeros(Nd)) ) # Zero pad 
            VA_orig.append(va)
        for idn in range(Nd):
            vd = vd_orig[:, idn] 
            vd = np.concatenate( (np.zeros(Na), vd) )
            VD_orig.append(vd)
        
        # Convert the list of column vectors to a matrix
        VA_orig = np.array(VA_orig).T 
        VD_orig = np.array(VD_orig).T

        # Combine the fragment eigenvectors to a block matrix
        V_all = np.concatenate( (VA_orig, VD_orig), axis = 1)

        # Get overlaps of the diabatic states, in the original (NAO) basis with S.
        dim_tot = V_all.shape[0]
        overlaps = np.zeros(V_all.shape)
        for i in range(dim_tot):
            for j in range(dim_tot):
                overlaps[i,j] = V_all[:, i] @ S_dimer @ V_all[:, j] 
        
        # Dimer Hamiltonian in the diabatic basis (Still non-orthogonalized)
        H_dimer_prime = V_all.T @ H_dimer @ V_all

        # Store the transformed Hamiltonian and overlap
        self.HS_diabats[xc][basis][self.dimer_distance]["H_dimer_prime"] = H_dimer_prime
        self.HS_diabats[xc][basis][self.dimer_distance]["overlaps"] = overlaps
        self.HS_diabats[xc][basis][self.dimer_distance]["NaNd"] = Na, Nd

    def get_couplings(self, xc: str, basis: str, iacc: int, idon: int):

        try:
            if self.HS_diabats[xc][basis][self.dimer_distance] and self.HS_diabats[xc][basis][self.dimer_distance]:
                print("Getting couplings directly from stored diabats")
            else:
                raise KeyError
        except KeyError:
            print(f"No diabats are stored for: \n Functional: {xc} \n Basis: {basis} \n Dimer distance {self.dimer_distance} \n Calculating HS matrices first")
            self.calculate_couplings_matrix(xc, basis)
        
        H_dimer_prime = self.HS_diabats[xc][basis][self.dimer_distance]["H_dimer_prime"]
        overlaps = self.HS_diabats[xc][basis][self.dimer_distance]["overlaps"]
        Na, Nd = self.HS_diabats[xc][basis][self.dimer_distance]["NaNd"]

        # Now identify matrix elements between the diabats within the dimer
        H22 = np.array( [H_dimer_prime[iacc, iacc], H_dimer_prime[iacc, Na + idon],
                 H_dimer_prime[Na + idon, iacc], H_dimer_prime[Na + idon, Na + idon]]).reshape(2, 2)
        
        # The overlap matrix of the diabatic states which are localized on the fragments
        S22 = np.array([overlaps[iacc, iacc], overlaps[iacc, Na + idon],
                        overlaps[Na + idon, iacc], overlaps[Na + idon, Na + idon]]).reshape(2, 2)
        
        # Lowdin biorthogonalization of the 2x2 system.
        X22 = scipy.linalg.sqrtm(scipy.linalg.inv(S22))
        Heff22_lowdin = X22.T @ H22 @ X22

        Jab = H22[0,1]
        Sab = S22[0,1]
        ea = H22[0,0]
        eb = H22[1,1]

        Hab22 = ( Jab - 0.5 * Sab * (ea + eb) ) / (1 - Sab**2) # Baumeier, Valeev equation 10
        Hab_fo = Hab22

        # Effective site energies:
        ea = H22[0,0]
        ed = H22[1,1]
        Ea_eff = 0.5 * ( (ea + ed) - 2 * Jab*Sab + (ea - ed) * np.sqrt(1 - Sab**2) ) / (1 - Sab**2) 
        Ed_eff = 0.5 * ( (ea + ed) + 2 * Jab*Sab - (ea - ed) * np.sqrt(1 - Sab**2) ) / (1 - Sab**2) 
        
        print(f"Hab_fo: {Hab_fo} eV \nSite energy acceptor: {ea} eV \nSite energy donor: {ed} eV")
        return Hab_fo, Sab, ea, ed #, Ea_eff, Ed_eff


    def dump_HS_matrices(self, path="fodft/calc_dir/", name = None):
        """
        Dump the Hamiltonian and Overlap matrices to a pickle file
        """
        if name is not None:
            path = path + name
        else:
            path = path + f"{self.dimerobject.mol_name}_{self.dimer_distance}_HS_matrices.pickle"

        to_be_dumped = self._convert_to_dict(self.HS_matrices)

        pickle.dump(to_be_dumped, open(path, "wb"))
        print("Dumped HS matrices to", path)

    def _convert_to_dict(self, d: dict):
        if isinstance(d, defaultdict):
            d = {k: self._convert_to_dict(v) for k, v in d.items()}

        return d

    def _update_directories(self, xc: str, basis: str, **kwargs):
        """
        Update the directories for the calculations
        """

        os.makedirs(f"fodft/calc_dir/{self.dimerobject.mol_name}/xc_{xc}/basis_{basis}/dist_{self.dimer_distance}", exist_ok=True)
        # txt_path = f"fodft/calc_dir/{self.dimerobject.mol_name}/xc_{xc}/basis_{basis}/dist_{self.dimer_distance}/{self.dimerobject.mol_name}_{self.frags[i]}.txt"
