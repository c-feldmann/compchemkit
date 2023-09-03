from typing import Iterable, Optional

import rdkit.Chem as Chem
from rdkit.Chem import FilterCatalog
import multiprocessing

from compchemkit.utils.parallel import check_adapt_n_jobs


class PainsFilter:
    _n_jobs: int

    def __init__(self, n_jobs: int = -1):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.filter: FilterCatalog = FilterCatalog.FilterCatalog(params)
        self.n_jobs = n_jobs

    @property
    def n_jobs(self) -> int:
        """Returns the number of cores used during filtering."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_cores: int) -> None:
        """Sets the number of cores used during filtering.

        Parameters
        ----------
        n_cores: int
            Number of requested cores.

        Returns
        -------
        None
        """
        self._n_jobs = check_adapt_n_jobs(n_cores)

    def check_smiles(self, smiles: str) -> Optional[bool]:
        """Check a smiles if they match any PAINS filter.

        Parameters
        ----------
        smiles: str
            SMILES representations of molecule.

        Returns
        -------
        Optional[bool]
            True: Contains PAINS substructure
            False: No PAINS substructure detected
            None: Invalid molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        else:
            return bool(self.filter.HasMatch(mol))

    def check_smiles_list(self, smiles_list: Iterable[str]) -> list[Optional[bool]]:
        """Check a list of smiles if they match any PAINS filter.

        Parameters
        ----------
        smiles_list: Iterable[str]
            Iterable of SMILES representations.

        Returns
        -------
        list[Optional[bool]]
            True: Contains PAINS substructure
            False: No PAINS substructure detected
            None: Invalid molecule.
        """
        if self._n_jobs == 1:
            return [self.check_smiles(smi) for smi in smiles_list]
        else:
            pool = multiprocessing.Pool(processes=self.n_jobs)
            return list(pool.map(self.check_smiles, smiles_list))
