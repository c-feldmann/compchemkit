from __future__ import annotations
import abc
from typing import Iterable, Optional, Self, NamedTuple
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FilterCatalog
from scipy import sparse

from compchemkit.utils.molecule_validity import construct_check_mol_list
from compchemkit.utils.matrix import generate_matrix_from_item_list


class AtomEnvironment(NamedTuple):
    """ "A Class to store environment-information for fingerprint features"""
    environment_atoms: set[int]


class CircularAtomEnvironment(AtomEnvironment):
    """ "A Class to store environment-information for morgan-fingerprint features"""
    central_atom: int
    radius: int

    def __new__(cls, central_atom: int, radius: int, environment_atoms: set[int]) -> CircularAtomEnvironment:
        self = super(CircularAtomEnvironment, cls).__new__(cls, environment_atoms)
        self.central_atom = central_atom
        self.radius = radius
        return self


class Fingerprint(abc.ABC):
    """A metaclass representing all fingerprint subclasses."""

    n_jobs: int
    _n_bits: int

    def __init__(self, n_jobs: int) -> None:
        self.n_jobs = n_jobs

    @property
    def n_bits(self) -> int:
        """Return number of bits, which is the length of the feature vector."""
        return self._n_bits

    @abc.abstractmethod
    def fit(self, mol_obj_list: list[Chem.Mol]) -> Self:
        """Fit fingerprint generator to input.

        Parameters
        ----------
        mol_obj_list: Iterable[Chem.Mol]
            A collection of molecules.

        Returns
        -------
        Self
            Fitted fingerprint generator.
        """

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        """Fit Fingerprint and transform the input.

        Parameters
        ----------
        mol_obj_list: list[Chem.Mol]
            List of molecules to which the object is fitted.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint for molecules.
        """
        self.fit(mol_obj_list)
        return self.transform(mol_obj_list)

    def transform(self, mol_obj_list: Iterable[Chem.Mol]) -> sparse.csr_matrix:
        """Transform iterable of RDKit Molecules to fingerprint.

        Parameters
        ----------
        mol_obj_list: Iterable[Chem.Mol]
            Iterable of molecules to transform.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint as sparse matrix.
        """
        if self.n_jobs == 1:
            bit_dict_list = (self._transform_mol(mol) for mol in mol_obj_list)
            return generate_matrix_from_item_list(bit_dict_list, self.n_bits)

        with Pool(self.n_jobs) as pool:
            bit_dict_iterator = pool.imap(self._transform_mol, mol_obj_list)
            return generate_matrix_from_item_list(bit_dict_iterator, self.n_bits)

    @abc.abstractmethod
    def _transform_mol(self, mol: Chem.Mol) -> dict[int, int]:
        """Calculate fingerprint of molecule and return as dict.

        Parameters
        ---------
        mol: Chem.Mol
            RDKit Molecule

        Returns
        -------
        dict[int, int]
            Key: item position in vector
            Value: item value
        """

    def fit_smiles(self, smiles_list: list[str]) -> Self:
        """Create a list of RDKit molecules from SMILES list and apply .fit method.

        Parameters
        ----------
        smiles_list: Iterable[str]
            List of SMILES strings.

        Returns
        -------
        Self
            Fitted object.
        """
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)
        return self

    def fit_transform_smiles(self, smiles_list: list[str]) -> sparse.csr_matrix:
        """Apply .fit_transform method based on SMILES list .

        Parameters
        ----------
        smiles_list: Iterable[str]
            List of SMILES strings.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint as sparse matrix.
        """
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: Iterable[str]) -> sparse.csr_matrix:
        """Apply .transform method from a list of smles.

        Parameters
        ----------
        smiles_list: Iterable[str]
            List of SMILES strings.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint as sparse matrix.
        """
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)


class _MorganFingerprint(Fingerprint):
    _radius: int

    def __init__(self, radius: int = 2, use_features: bool = False, n_jobs: int = 1):
        super().__init__(n_jobs=n_jobs)
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(
                f"Number of bits has to be a positive integer! (Received: {radius})"
            )

    def __len__(self) -> int:
        """Return length of the feature vector."""
        return self.n_bits

    @property
    def radius(self) -> int:
        """Return size of radius used to derive features."""
        return self._radius

    @property
    def use_features(self) -> bool:
        """Return if atoms are represented as feature."""
        return self._use_features

    @abc.abstractmethod
    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        """Return a dictionary where of features and corresponding environments, listing central atom and radius.

        Parameters
        ----------
        mol_obj: Chem.Mol
            RDKit molecule which is explained.

        Returns
        -------
        dict[int, list[tuple[int, int]]]
            Key: feature position
            Value: list of matching environments.
                Matching environments are given as tuple of central atom id and radius.
        """

    def bit2atom_mapping(
        self, mol_obj: Chem.Mol
    ) -> dict[int, list[CircularAtomEnvironment]]:
        """Map list of matching CircularAtomEnvironment to each bit.

        Bits may occure multiple times, and hence each item in the list represents one occurrence.
        Items are AtomEnvironment objects containing indices of corresponding atoms.

        Parameters
        ----------
        mol_obj: Chem.Mol
            Molecule to explain.

        Returns
        -------
        dict[int, list[CircularAtomEnvironment]]
            Key: bit-position
            Value: list of encoded CircularAtomEnvironment for the bit.
        """
        bit2atom_dict = self.explain_rdmol(mol_obj)
        result_dict = defaultdict(list)

        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():
            for central_atom, radius in matches:
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
    def __init__(
        self,
        n_bits: int = 2048,
        radius: int = 2,
        use_features: bool = False,
        n_jobs: int = 1,
    ):
        """Initialize fingerprint generation method for folded morgan fingerprint.

        Parameters
        ----------
        n_bits: int
            Size of returned bit-vector.
        radius: int
            Maximum radius of derived circular features.
        use_features: bool
            Encode Atoms based on their features.
        n_jobs:
            Number of cores to use.

        Returns
        -------
        None
        """
        super().__init__(radius=radius, use_features=use_features, n_jobs=n_jobs)
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(
                f"Number of bits has to be a positive integer! (Received: {n_bits})"
            )

    def fit(self, mol_obj_list: list[Chem.Mol]) -> Self:
        """Fit fingerprint generator to input.

        Does nothing.

        Parameters
        ----------
        mol_obj_list: Iterable[Chem.Mol]
            A collection of molecules.

        Returns
        -------
        Self
            Fitted fingerprint generator.
        """
        return self

    def _transform_mol(self, mol: Chem.Mol) -> dict[int, int]:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, useFeatures=self._use_features, nBits=self._n_bits
        )
        return {bit: 1 for bit in fp.GetOnBits()}

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        """Return a dictionary where of features and corresponding environments, listing central atom and radius.

        Parameters
        ----------
        mol_obj: Chem.Mol
            RDKit molecule which is explained.

        Returns
        -------
        dict[int, list[tuple[int, int]]]
            Key: feature position
            Value: list of matching environments.
                Matching environments are given as tuple of central atom id and radius.
        """
        bi: dict[int, list[tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol_obj,
            self.radius,
            useFeatures=self._use_features,
            bitInfo=bi,
            nBits=self._n_bits,
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

    _bit_mapping: Optional[dict[int, int]]
    _counted: bool
    ignore_unknown: bool

    def __init__(
        self,
        counted: bool = False,
        radius: int = 2,
        use_features: bool = False,
        ignore_unknown: bool = False,
        n_jobs: int = 1,
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
        super().__init__(radius=radius, use_features=use_features, n_jobs=n_jobs)
        self._bit_mapping = None

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
        """Return copy of dict used for mapping feature hash to bit position.

        Raises
        ------
        AttributeError
            When object is not fitted
        """
        if self._bit_mapping is None:
            raise AttributeError("Attribute not set. Please call fit first.")
        return self._bit_mapping

    def fit(self, mol_obj_list: list[Chem.Mol]) -> Self:
        """Fit fingerprint generator to input.

        Parameters
        ----------
        mol_obj_list: Iterable[Chem.Mol]
            A collection of molecules.

        Returns
        -------
        Self
            Fitted fingerprint generator.
        """
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        self._create_mapping(mol_iterator)
        return self

    def _gen_features(self, mol_obj: Chem.Mol) -> dict[int, int]:
        """Return a dict, where the key is the feature-hash and the value is the count."""
        return dict(
            AllChem.GetMorganFingerprint(
                mol_obj, self.radius, useFeatures=self.use_features
            ).GetNonzeroElements()
        )

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict[int, list[tuple[int, int]]]:
        """Return a dictionary where of features and corresponding environments, listing central atom and radius.

        Parameters
        ----------
        mol_obj: Chem.Mol
            RDKit molecule which is explained.

        Returns
        -------
        dict[int, list[tuple[int, int]]]
            Key: feature position
            Value: list of matching environments.
                Matching environments are given as tuple of central atom id and radius.
        """
        bi: dict[int, list[tuple[int, int]]] = dict()
        _ = AllChem.GetMorganFingerprint(
            mol_obj, self.radius, useFeatures=self.use_features, bitInfo=bi
        )
        bit_info = {self.bit_mapping[k]: v for k, v in bi.items()}
        return bit_info

    def fit_transform(self, mol_obj_list: list[Chem.Mol]) -> sparse.csr_matrix:
        """Fit Fingerprint and transform the input.

        Parameters
        ----------
        mol_obj_list: list[Chem.Mol]
            List of molecules to which the object is fitted.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint for molecules.
        """
        mol_fp_list = [self._gen_features(mol_obj) for mol_obj in mol_obj_list]
        self._create_mapping(mol_fp_list)
        new_fp_list = [self._map_features_dict(f_dict) for f_dict in mol_fp_list]
        return generate_matrix_from_item_list(new_fp_list, self.n_bits)

    def _transform_mol(self, mol: Chem.Mol) -> dict[int, int]:
        feature_dict = self._gen_features(mol)
        return self._map_features_dict(feature_dict)

    def _map_features_dict(self, feature_hash_dict: dict[int, int]) -> dict[int, int]:
        unknown_features = set(feature_hash_dict.keys()) - set(self.bit_mapping.keys())
        if unknown_features:
            if self.ignore_unknown:
                for unknown_feature in unknown_features:
                    feature_hash_dict.pop(unknown_feature)
            else:
                raise ValueError(f"Unknown feature hashs: {unknown_features}")
        if self.counted:
            return {self.bit_mapping[f]: c for f, c in feature_hash_dict.items()}
        return {self.bit_mapping[f]: 1 for f, c in feature_hash_dict.items()}

    def _create_mapping(self, molecule_features: Iterable[dict[int, int]]) -> None:
        unraveled_features = [f for f_list in molecule_features for f in f_list.keys()]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(
            unique_features, key=lambda f: (feature_hash_dict[f], f), reverse=True
        )
        self._bit_mapping = {feature: idx for idx, feature in enumerate(feature_order)}
        self._n_bits = len(self._bit_mapping)


class MACCS(Fingerprint):
    """Calculates the MACCS fingerprint for given molecules.

    The MACCS fingerprint consists of 166 predefined substructures.
    """

    def __init__(self, n_jobs: int = 1) -> None:
        """Initialize MACCS Key fingerprint generator.

        Parameters
        ----------
        n_jobs: number of cores to use.

        Returns
        -------
        None
        """
        super().__init__(n_jobs=n_jobs)
        self._n_bits = 166

    def fit(self, mol_obj_list: Iterable[Chem.Mol]) -> Self:
        """Fit fingerprint generator to input.

        Does nothing as this class requires no fitting.

        Parameters
        ----------
        mol_obj_list: Iterable[Chem.Mol]
            A collection of molecules.

        Returns
        -------
        Self
            Fitted fingerprint generator.
        """
        return self

    def _transform_mol(self, mol: Chem.Mol) -> dict[int, int]:
        maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
        return {bit: 1 for bit in maccs_fp.GetOnBits()}

    def transform(self, mol_obj_list: Iterable[Chem.Mol]) -> sparse.csr_matrix:
        """Transform iterable of RDKit Molecules to MACCs key fingerprint.

        Parameters
        ----------
        mol_obj_list: Iterable[Chem.Mol]
            Iterable of molecules to transform.

        Returns
        -------
        sparse.csr_matrix
            Fingerprint as sparse matrix.
        """
        r_matrix = super().transform(mol_obj_list)
        # Remove first colum as this is always zero
        # https://github.com/rdkit/rdkit/blob/b208da471f8edc88e07c77ed7d7868649ac75100/Code/GraphMol/Fingerprints/MACCS.h#L17
        if r_matrix[:, 0].sum() != 0:
            raise AssertionError(
                "First colum should always be zero. Please file a bug report!"
            )
        return r_matrix[:, 1:]


class FragmentFingerprint(Fingerprint):
    def __init__(self, substructure_list: list[str], n_jobs: int = 1) -> None:
        """Fingerprint from a list of substructures.

        Parameters
        ----------
        substructure_list: list[str]
            List of SMARTS representations.
        n_jobs: int
            Number of core to use.

        Returns
        -------
        None
        """
        super().__init__(n_jobs=n_jobs)
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

    def _transform_mol(self, mol: Chem.Mol) -> dict[int, int]:
        feature_dict = {
            int(match.GetDescription()): 1 for match in self._filter.GetMatches(mol)
        }
        return feature_dict

    def fit(self, mol_obj_list: list[Chem.Mol]) -> Self:
        """Fit generator to the input.

        Does nothing.

        Parameters
        ----------
        mol_obj_list: list[Chem.Mol]
            List of molecules to which the object is fitted

        Returns
        -------
        Self:
            Fitted generator
        """
        return self

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> dict[int, list[AtomEnvironment]]:
        """Map list of matching AtomEnvironments to each bit.

        Bits may occure multiple times, and hence each item in the list represents one occurrence.
        Items are AtomEnvironment objects containing indices of corresponding atoms.

        Parameters
        ----------
        mol_obj: Chem.Mol
            Molecule to explain.

        Returns
        -------
        dict[int, list[AtomEnvironment]]
            Key: bit-position
            Value: list of encoded AtomsEnvironments for the bit.
        """
        present_bits = self._transform_mol(mol_obj)
        bit2atom_dict = defaultdict(list)
        for bit in present_bits:
            bit_smarts_obj = self._substructure_obj_list[bit]
            matches = mol_obj.GetSubstructMatches(bit_smarts_obj)
            for match in matches:
                atom_env = AtomEnvironment(match)
                bit2atom_dict[bit].append(atom_env)

        # Transforming defaultdict to dict
        return dict(bit2atom_dict.items())
