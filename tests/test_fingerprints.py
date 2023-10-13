import unittest

import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from rdkit import Chem
from scipy import sparse

from compchemkit.fingerprints import (
    UnfoldedMorganFingerprint,
    FragmentFingerprint,
)
from compchemkit.utils.molecule_validity import construct_check_mol_list

test_folder = os.path.dirname(__file__)
smiles_df = pd.read_csv(f"{test_folder}/test_data/test_smiles.tsv")
smiles_list = smiles_df["SMILES"].to_list()


class ConstructingFingerprints(unittest.TestCase):
    def test_independence_of_constructing(self) -> None:
        mol_obj_list = construct_check_mol_list(smiles_list)
        ecfp2_1 = UnfoldedMorganFingerprint()
        fp1 = ecfp2_1.fit_transform(mol_obj_list)
        ecfp2_2 = UnfoldedMorganFingerprint()
        ecfp2_2.fit(mol_obj_list)
        fp2 = ecfp2_2.transform(mol_obj_list)
        self.assertEqual((fp1 != fp2).nnz, 0)

    def test_independence_of_constructing_parallel(self) -> None:
        mol_obj_list = construct_check_mol_list(smiles_list)
        # Fingerprint 1
        ecfp2_1 = UnfoldedMorganFingerprint(n_jobs=2)
        fp1 = ecfp2_1.fit_transform(mol_obj_list)
        # Fingerprint 2
        ecfp2_2 = UnfoldedMorganFingerprint(n_jobs=2)
        ecfp2_2.fit(mol_obj_list)
        fp2 = ecfp2_2.transform(mol_obj_list)
        # Fingerprint 3
        ecfp2_3 = UnfoldedMorganFingerprint(n_jobs=1)
        fp3 = ecfp2_3.fit_transform(mol_obj_list)
        # Compare for equal
        self.assertEqual((fp1 != fp2).nnz, 0)
        self.assertEqual((fp1 != fp3).nnz, 0)

    def test_substructure_fp(self) -> None:
        smarts_list = ["[#6]", "[#7]", "[#8]"]
        frag_fingerprint = FragmentFingerprint(smarts_list)
        fp = frag_fingerprint.transform(construct_check_mol_list(smiles_list))
        expected_fp: npt.NDArray[np.int_] = np.zeros(
            (len(smiles_list), len(smarts_list)), dtype=int
        )
        for i, smart in enumerate(smarts_list):
            smart_obj = Chem.MolFromSmarts(smart)
            for j, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol.HasSubstructMatch(smart_obj):
                    expected_fp[j, i] = 1
        expected_fp_sparse = sparse.csr_matrix(expected_fp)
        self.assertEqual((fp != expected_fp_sparse).nnz, 0)


if __name__ == "__main__":
    unittest.main()
