from __future__ import annotations
import abc
from typing import Iterable, Optional
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FilterCatalog
from scipy import sparse

from compchemkit.supporting_functions import construct_check_mol_list, check_correct_num_cores

from tqdm.auto import tqdm


class AtomEnvironment:
    """ "A Class to store environment-information for fingerprint features"""

    def __init__(self, environment_atoms: set[int]):
        self.environment_atoms = environment_atoms  # set of all atoms within radius


class CircularAtomEnvironment(AtomEnvironment):
    """ "A Class to store environment-information for morgan-fingerprint features"""

    def __init__(self, central_atom: int, radius: int, environment_atoms: set[int]):
        super().__init__(environment_atoms)
        self.central_atom = central_atom
        self.radius = radius


class Fingerprint(abc.ABC):
    """A metaclass representing all fingerprint subclasses."""
    @property
    @abc.abstractmethod
    def n_bits(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    def fit_smiles(self, smiles_list: list[str]) -> Fingerprint:
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)
        return self

    def fit_transform_smiles(self, smiles_list: list[str]) -> sparse.csr_matrix:
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: list[str]) -> sparse.csr_matrix:
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)


class _MorganFingerprint(Fingerprint):
    _n_bits: Optional[int]

    def __init__(self, radius: int = 2, use_features: bool = False):
        super().__init__()
        self._n_bits = None
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {radius})")

    def __len__(self) -> int:
        return self.n_bits

    @property
    def n_bits(self) -> int:
        if self._n_bits is None:
            raise ValueError("Number of bits is undetermined!")
        return self._n_bits

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def use_features(self) -> bool:
        return self._use_features

    @abc.abstractmethod
    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        raise NotImplementedError

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> dict[int, list[CircularAtomEnvironment]]:
        bit2atom_dict = self.explain_rdmol(mol_obj)
        result_dict = defaultdict(list)

        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():  # type: int, list[tuple[int, int]]
            for central_atom, radius in matches:  # type: int, int
                if radius == 0:
                    result_dict[bit].append(
                        CircularAtomEnvironment(central_atom, radius, {central_atom})
                    )
                    continue
                env = Chem.FindAtomEnvironmentOfRadiusN(mol_obj, radius, central_atom)
                amap: dict[int, int] = dict()
                _ = Chem.PathToSubmol(mol_obj, env, atomMap=amap)
                env_atoms = amap.keys()
                assert central_atom in env_atoms
                result_dict[bit].append(
                    CircularAtomEnvironment(central_atom, radius, set(env_atoms))
                )

        # Transforming default dict to dict
        return {k: v for k, v in result_dict.items()}


class FoldedMorganFingerprint(_MorganFingerprint):
    def __init__(self, n_bits: int = 2048, radius: int = 2, use_features: bool = False, n_cores: int = 1):
        super().__init__(radius=radius, use_features=use_features)
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {n_bits})")

    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        pass

    def transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        fingerprints = []
        for mol in mol_obj_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, useFeatures=self._use_features, nBits=self._n_bits
            )
            fingerprints.append(sparse.csr_matrix(fp))
        return sparse.vstack(fingerprints)

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        bi: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol_obj, self.radius, useFeatures=self._use_features, bitInfo=bi, nBits=self._n_bits
        )
        return bi


class UnfoldedMorganFingerprint(_MorganFingerprint):
    """Transforms smiles-strings or molecular objects into unfolded bit-vectors based on Morgan-fingerprints [1].
    Features are mapped to bits based on the amount of molecules they occur in.

    Long version:
        Circular fingerprints do not have a unique mapping to a bit-vector, therefore the features are mapped to the
        vector according to the number of molecules they occur in. The most occurring feature is mapped to bit 0, the
        second most feature to bit 1 and so on...

        Weak-point: features not seen in the fit method are not mappable to the bit-vector and therefore cause an error.

    References:
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """

    def __init__(
        self,
        counted: bool = False,
        radius: int = 2,
        use_features: bool = False,
        ignore_unknown: bool = False,
    ):
        """Initializes the class
        Parameters
        ----------
        counted: bool
            False: bits are binary: on if present in molecule, off if not present
            True: bits are positive integers and give the occurrence of their respective features in the molecule
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            Instead of atoms, features are encoded in the fingerprint. [2]

        References
        ----------
            [1] https://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
            [2] https://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """
        super().__init__(radius=radius, use_features=use_features)
        self._bit_mapping: Optional[dict[int, int]] = None

        if not isinstance(counted, bool):
            raise TypeError("The argument 'counted' must be a bool!")
        self._counted = counted

        if not isinstance(ignore_unknown, bool):
            raise TypeError("The argument 'ignore_unknown' must be a bool!")
        self.ignore_unknown = ignore_unknown

    @property
    def counted(self) -> bool:
        """Returns the bool value for enabling counted fingerprint."""
        return self._counted

    @property
    def bit_mapping(self) -> dict[int, int]:
        if self._bit_mapping is None:
            raise AttributeError("Attribute not set. Please call fit first.")
        return self._bit_mapping.copy()

    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        self._create_mapping(mol_iterator)

    def _gen_features(self, mol_obj: Chem.Mol) -> dict[int, int]:
        """Return a dict, where the key is the feature-hash and the value is the count."""
        return AllChem.GetMorganFingerprint(
            mol_obj, self.radius, useFeatures=self.use_features
        ).GetNonzeroElements()

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        bi: dict[int, list[tuple[int, int]]] = dict()
        _ = AllChem.GetMorganFingerprint(
            mol_obj, self.radius, useFeatures=self.use_features, bitInfo=bi
        )
        bit_info = {self.bit_mapping[k]: v for k, v in bi.items()}
        return bit_info

    def explain_smiles(self, smiles: str) -> dict[int, list[tuple[int, int]]]:
        return self.explain_rdmol(Chem.MolFromSmiles(smiles))

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        mol_fp_list = [self._gen_features(mol_obj) for mol_obj in mol_obj_list]
        self._create_mapping(mol_fp_list)
        return self._transform(mol_fp_list)

    def transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_iterator)

    def _map_features(self, feature_hash_list: list[int]) -> list[int]:
        feature_pos_list = list(map(self.bit_mapping.get, feature_hash_list))
        if None in feature_pos_list:
            if not self.ignore_unknown:
                none_hash_list = [f_hash for f_hash, f_pos in zip(feature_hash_list, feature_pos_list) if f_pos is None]
                raise KeyError(f"Following features are not known: {none_hash_list}")
            else:
                feature_pos_list = [feature_pos for feature_pos in feature_pos_list if feature_pos is not None]
        return feature_pos_list

    def _transform(
        self,
        mol_fp_list: Iterable[dict[int, int]],
    ) -> sparse.csr_matrix:
        data: list[int] = []
        rows: list[int] = []
        cols: list[int] = []
        n_col = 0
        if self._counted:
            for i, mol_fp in enumerate(mol_fp_list):  # type: int, dict[int, int]
                features = list(mol_fp.keys())
                counts = [mol_fp[f] for f in features]
                data.extend(counts)
                rows.extend(self._map_features(features))
                cols.append(i)
                n_col += 1
        else:
            for i, mol_fp in enumerate(mol_fp_list):
                data.extend([1] * len(mol_fp))
                feature_hash_list = list(mol_fp.keys())
                rows.extend(self._map_features(feature_hash_list))
                cols.extend([i] * len(mol_fp))
                n_col += 1
        return sparse.csr_matrix((data, (cols, rows)), shape=(n_col, self.n_bits))

    def _create_mapping(self, molecule_features: Iterable[dict[int, int]]) -> None:
        unraveled_features = [f for f_list in molecule_features for f in f_list.keys()]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(
            unique_features, key=lambda f: (feature_hash_dict[f], f), reverse=True
        )
        # self._bit_mapping = bidict(zip(feature_order, range(len(feature_order))))
        self._bit_mapping = dict({feature: idx for idx, feature in enumerate(feature_order)})
        self._n_bits = len(self._bit_mapping)


class MACCS(Fingerprint):
    def __init__(self) -> None:
        super().__init__()

    @property
    def n_bits(self) -> int:
        return 166

    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        pass

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        fingerprints = []
        for mol_obj in mol_obj_list:
            fingerprints.append(sparse.csr_matrix(AllChem.GetMACCSKeysFingerprint(mol_obj)))
        r_matrix = sparse.vstack(fingerprints)
        # rdkit maccs indexing starts at 1 not at zero, remove first, always empty line
        # https://github.com/rdkit/rdkit/blob/b208da471f8edc88e07c77ed7d7868649ac75100/Code/GraphMol/Fingerprints/MACCS.h#L17
        assert r_matrix[:, 0].sum() == 0
        return r_matrix[:, 1:]


class FragmentFingerprint(Fingerprint):
    def __init__(self, substructure_list: list[str]):
        super(FragmentFingerprint, self).__init__()
        self._substructure_list = substructure_list
        self._substructure_obj_list = []

        self._filter = FilterCatalog.FilterCatalog()
        self._n_bits: int = len(self._substructure_list)

        for i, substructure in enumerate(self._substructure_list):
            # Validating Smarts
            smarts_obj = Chem.MolFromSmarts(substructure)
            if smarts_obj is None:
                raise ValueError(f"Invalid SMARTS pattern: {substructure}")
            self._substructure_obj_list.append(smarts_obj)

            # Adding pattern to the filter catalogue
            pattern = FilterCatalog.SmartsMatcher(f"Pattern {i}", substructure, 1)
            self._filter.AddEntry(FilterCatalog.FilterCatalogEntry(str(i), pattern))

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def _gen_features(self, mol_obj: Chem.Mol) -> list[int]:
        return [int(match.GetDescription()) for match in self._filter.GetMatches(mol_obj)]

    def _transform(self, mol_fp_list: Iterable[list[int]]) -> sparse.csr_matrix:
        data = []
        rows = []
        cols = []
        n_col = 0
        for i, mol_fp in enumerate(mol_fp_list):
            data.extend([1] * len(mol_fp))
            rows.extend(mol_fp)
            cols.extend([i] * len(mol_fp))
            n_col += 1
        return sparse.csr_matrix((data, (cols, rows)), shape=(n_col, self.n_bits))

    def fit(self, mol_obj_list: list[Chem.Mol]) -> None:
        pass

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        mol_feature_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_feature_iterator)

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> dict[int, list[AtomEnvironment]]:
        present_bits = self._gen_features(mol_obj)
        bit2atom_dict = defaultdict(list)
        for bit in present_bits:
            bit_smarts_obj = self._substructure_obj_list[bit]
            matches = mol_obj.GetSubstructMatches(bit_smarts_obj)
            for match in matches:
                atom_env = AtomEnvironment(match)
                bit2atom_dict[bit].append(atom_env)

        # Transforming defaultdict to dict
        return {k: v for k, v in bit2atom_dict.items()}


if __name__ == "__main__":
    # noinspection SpellCheckingInspection
    test_smiles_list = [
        "c1ccccc1",
        "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
        "c1cc(ccc1C2CCNCC2COc3ccc4c(c3)OCO4)F",
        "c1c(c2c(ncnc2n1C3C(C(C(O3)CO)O)O)N)C(=O)N",
        "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl",
        "CN(C)c1c2c(ncn1)n(cn2)C3C(C(C(O3)CO)NC(=O)C(Cc4ccc(cc4)OC)N)O",
        "CC12CCC(CC1CCC3C2CC(C4(C3(CCC4C5=CC(=O)OC5)O)C)O)O",
    ]
    test_mol_obj_list = construct_check_mol_list(test_smiles_list)

    ecfp2_1 = UnfoldedMorganFingerprint()
    fp1 = ecfp2_1.fit_transform(test_mol_obj_list)
    print(fp1.shape)

    ecfp2_folded = FoldedMorganFingerprint()
    fp2 = ecfp2_folded.fit_transform(test_mol_obj_list)
    print(fp2.shape)

    maccs = MACCS()
    maccs_fp = maccs.fit_transform(test_mol_obj_list)
    print(maccs_fp.shape)
