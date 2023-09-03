from typing import Iterable
import rdkit.Chem as Chem


def construct_check_mol_list(smiles_list: Iterable[str]) -> list[Chem.Mol]:
    """Transform a list of SMILES to RDKit molecules.

    Invalid encodings are collected and raised, once all mols are checked.

    Parameters
    ----------
    smiles_list: Iterable[str]
        list of SMILES representations.

    Returns
    -------
    list[Chem.Mol]
        Tranformed molecules.

    Raises
    ------
    ValueError
        In case of invalid molecular encodings.
    """
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles_list = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles_list.append(smiles)
        invalid_smiles_str = "\n".join(invalid_smiles_list)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles_str}")
    return mol_obj_list
