import unittest

import numpy as np
import scipy.sparse as sparse

from compchemkit.ml_utilities.fingerprints import UnfoldedMorganFingerprint
from compchemkit.ml_utilities.kernel import TanimotoKernel
from compchemkit.supporting_functions import construct_check_mol_list

# noinspection SpellCheckingInspection
smiles_list = ["c1ccccc1",
               "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
               "c1cc(ccc1C2CCNCC2COc3ccc4c(c3)OCO4)F",
               "c1c(c2c(ncnc2n1C3C(C(C(O3)CO)O)O)N)C(=O)N",
               "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl",
               "CN(C)c1c2c(ncn1)n(cn2)C3C(C(C(O3)CO)NC(=O)C(Cc4ccc(cc4)OC)N)O",
               "CC12CCC(CC1CCC3C2CC(C4(C3(CCC4C5=CC(=O)OC5)O)C)O)O",
               ]


class ConstructingFingerprints(unittest.TestCase):
    def test_independence_of_constructing(self):
        mol_obj_list = construct_check_mol_list(smiles_list)
        ecfp2_1 = UnfoldedMorganFingerprint()
        fp1 = ecfp2_1.fit_transform(mol_obj_list)
        ecfp2_2 = UnfoldedMorganFingerprint()
        ecfp2_2.fit(mol_obj_list)
        fp2 = ecfp2_2.transform(mol_obj_list)
        self.assertTrue((fp1 != fp2).nnz == 0)


class Kernel(unittest.TestCase):
    def test_sparse_kernel_simple_vectors(self):
        test_fingerprint1 = sparse.csr_matrix(np.array([[0, 0, 0, 1],
                                                        [0, 0, 1, 1],
                                                        [0, 1, 0, 0]]
                                                       )
                                              )
        test_fingerprint2 = sparse.csr_matrix(np.array([[0, 0, 0, 1],
                                                        [0, 0, 1, 1],
                                                        [0, 1, 1, 0],
                                                        [1, 0, 0, 0],
                                                        ]
                                                       )
                                              )
        expected_matrix = np.array([[1, 0.5, 0, 0],
                                    [0.5, 1, 1 / 3, 0],
                                    [0, 0, 0.5, 0]
                                    ])

        self.assertTrue(np.all(np.isclose(TanimotoKernel.similarity_from_sparse(test_fingerprint1, test_fingerprint2),
                                          expected_matrix)))

    def test_dense_kernel_simple_vectors(self):
        test_fingerprint1 = np.array([[0, 0, 0, 1],
                                      [0, 0, 1, 1],
                                      [0, 1, 0, 0]]
                                     )
        test_fingerprint2 = np.array([[0, 0, 0, 1],
                                      [0, 0, 1, 1],
                                      [0, 1, 1, 0],
                                      [1, 0, 0, 0],
                                      ]

                                     )
        expected_matrix = np.array([[1, 0.5, 0, 0],
                                    [0.5, 1, 1 / 3, 0],
                                    [0, 0, 0.5, 0]
                                    ])

        self.assertTrue(np.all(np.isclose(TanimotoKernel.similarity_from_dense(test_fingerprint1, test_fingerprint2),
                                          expected_matrix)))

    def test_real_fp_as_input(self):
        mol_obj_list = construct_check_mol_list(smiles_list)
        ecfp2_1 = UnfoldedMorganFingerprint()
        fp1 = ecfp2_1.fit_transform(mol_obj_list)
        sim_matrix = TanimotoKernel.similarity_from_sparse(fp1, fp1)
        self.assertEqual(sim_matrix.shape[0], sim_matrix.shape[1])
        self.assertEqual(sim_matrix.shape[0], len(mol_obj_list))
        self.assertTrue(np.all(np.isclose(sim_matrix.diagonal(), np.ones((len(mol_obj_list), 1)))))


if __name__ == '__main__':
    unittest.main()
