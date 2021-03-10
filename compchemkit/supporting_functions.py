from typing import *

import rdkit.Chem as Chem


def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list


def construct_check_mol(smiles: str) -> Chem.Mol:
    mol_obj = Chem.MolFromSmiles(smiles)
    if not mol_obj:
        raise ValueError(f"Following smiles are not valid: {smiles}")
    return mol_obj
