import os
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from compchemkit.fingerprints import UnfoldedMorganFingerprint
from compchemkit.kernel import similarity_from_dense, similarity_from_sparse
from compchemkit.utils.molecule_validity import construct_check_mol_list


test_folder = os.path.dirname(__file__)
smiles_df = pd.read_csv(f"{test_folder}/test_data/test_smiles.tsv")
smiles_list = smiles_df["SMILES"].to_list()


class Kernel(unittest.TestCase):
    def test_sparse_kernel_simple_vectors(self) -> None:
        test_fingerprint1 = sparse.csr_matrix(
            np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0]])
        )
        test_fingerprint2 = sparse.csr_matrix(
            np.array(
                [
                    [0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 0, 0, 0],
                ]
            )
        )
        expected_matrix = np.array([[1, 0.5, 0, 0], [0.5, 1, 1 / 3, 0], [0, 0, 0.5, 0]])

        self.assertTrue(
            np.all(
                np.isclose(
                    similarity_from_sparse(test_fingerprint1, test_fingerprint2),
                    expected_matrix,
                )
            )
        )

    def test_dense_kernel_simple_vectors(self) -> None:
        test_fingerprint1 = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0]])
        test_fingerprint2 = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 0],
            ]
        )
        expected_matrix = np.array([[1, 0.5, 0, 0], [0.5, 1, 1 / 3, 0], [0, 0, 0.5, 0]])

        self.assertTrue(
            np.all(
                np.isclose(
                    similarity_from_dense(test_fingerprint1, test_fingerprint2),
                    expected_matrix,
                )
            )
        )

    def test_real_fp_as_input(self) -> None:
        mol_obj_list = construct_check_mol_list(smiles_list)
        ecfp2_1 = UnfoldedMorganFingerprint()
        fp1 = ecfp2_1.fit_transform(mol_obj_list)
        sim_matrix = similarity_from_sparse(fp1, fp1)
        self.assertEqual(sim_matrix.shape[0], sim_matrix.shape[1])
        self.assertEqual(sim_matrix.shape[0], len(mol_obj_list))
        self.assertTrue(
            np.all(np.isclose(sim_matrix.diagonal(), np.ones((len(mol_obj_list), 1))))
        )


if __name__ == '__main__':
    unittest.main()
