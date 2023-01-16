import multiprocessing
import rdkit.Chem as Chem


def construct_check_mol_list(smiles_list: list[str]) -> list[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles_list = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles_list.append(smiles)
        invalid_smiles_str = "\n".join(invalid_smiles_list)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles_str}")
    return mol_obj_list


def construct_check_mol(smiles: str) -> Chem.Mol:
    mol_obj = Chem.MolFromSmiles(smiles)
    if not mol_obj:
        raise ValueError(f"Following smiles are not valid: {smiles}")
    return mol_obj


def check_correct_num_cores(n_cores: int):
    if not isinstance(n_cores, int):
        raise TypeError(f"n_cores must be an int. Received: {type(n_cores)}")
    if n_cores == 1:
        return n_cores

    try:
        available_cpus = multiprocessing.cpu_count()
        if n_cores == -1:
            return available_cpus
        elif n_cores <= available_cpus:
            return n_cores
        else:
            print(f"More cores than available ({available_cpus}) requested! Falling back to {available_cpus}!")
            return available_cpus

    except NotImplementedError:
        print("Multiprocessing not supported. Falling back to single process!")
        return 1
