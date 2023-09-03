from typing import List, Optional

import rdkit.Chem as Chem
from rdkit.Chem import FilterCatalog
import multiprocessing


class PainsFilter:
    def __init__(self, n_cores: int = -1):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.filter: FilterCatalog = FilterCatalog.FilterCatalog(params)
        self.n_cores: int = n_cores

    @property
    def n_cores(self) -> int:
        """Returns the number of cores used during filtering."""
        return self._n_cores

    @n_cores.setter
    def n_cores(self, n_cores: int) -> None:
        """Sets the number of cores used during filtering.

        Parameters
        ----------
        n_cores: int
            Number of requested cores.

        Returns
        -------
        None
        """
        if not isinstance(n_cores, int):
            raise TypeError(f"n_cores must be an int. Received: {type(n_cores)}")

        if n_cores == 1:
            self._n_cores = n_cores
        else:
            try:
                available_cpus = multiprocessing.cpu_count()

                if n_cores == -1:
                    self._n_cores = available_cpus
                elif n_cores <= available_cpus:
                    self._n_cores = n_cores
                else:
                    print(
                        f"More cores than available requested! Falling back to {available_cpus}"
                    )
                    self._n_cores = available_cpus

            except NotImplementedError:
                print("multiprocessing not supported. Falling back to single process!")
                self._n_cores = 1  # fall_back solution

    def check_smiles(self, smiles: str) -> Optional[bool]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        else:
            return self.filter.HasMatch(mol)

    def check_smiles_list(self, smiles_list: List[str]) -> List[Optional[bool]]:
        if self.n_cores == 1:
            return [self.check_smiles(smi) for smi in smiles_list]
        else:
            pool = multiprocessing.Pool(processes=self.n_cores)
            return list(pool.map(self.check_smiles, smiles_list))
