"""Functions for filtering compounds."""

from typing import Iterable

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import FilterCatalog

from compchemkit.utils.parallel import check_adapt_n_jobs


class PainsFilter:
    """Class for removing compounds containing PAINS substructures."""

    _n_jobs: int

    def __init__(self, n_jobs: int = -1) -> None:
        """Initialize the PainsFilter object.

        Parameters
        ----------
        n_jobs: int, default: -1
            Number of workers to use.
        """
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.filter: FilterCatalog = FilterCatalog.FilterCatalog(params)
        self.n_jobs = n_jobs

    @property
    def n_jobs(self) -> int:
        """Return the number of cores used during filtering."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_cores: int) -> None:
        """Set the number of cores used during filtering.

        Parameters
        ----------
        n_cores: int
            Number of requested cores.
        """
        self._n_jobs = check_adapt_n_jobs(n_cores)

    def check_smiles(self, smiles: str) -> bool | None:
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
        return bool(self.filter.HasMatch(mol))

    def check_smiles_list(self, smiles_list: Iterable[str]) -> list[bool | None]:
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

        parallel = Parallel(n_jobs=self.n_jobs)
        return parallel(delayed(self.check_smiles)(smi) for smi in smiles_list)
