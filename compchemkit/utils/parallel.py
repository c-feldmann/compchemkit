import multiprocessing
import warnings


def check_adapt_n_jobs(n_jobs: int) -> int:
    """Check requested number of cores with available number of chores.

    Parameters
    ----------
    n_jobs: int
        Number of requested cores

    Returns
    -------
    int
        Number of cores to assign.
    """
    if not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs must be an int. Received: {type(n_jobs)}")

    if n_jobs == 1:
        return 1

    try:
        available_cpus = multiprocessing.cpu_count()

        if n_jobs == -1:
            return available_cpus
        if n_jobs <= available_cpus:
            return n_jobs
        else:
            warnings.warn(
                f"More cores than available requested! Falling back to {available_cpus}"
            )
            return available_cpus

    except NotImplementedError:
        warnings.warn("multiprocessing not supported. Falling back to single process!")
        return 1
