#!/usr/bin/env python3

import numpy as np

from pathlib import Path
import numpy.typing as npt
from typing import Literal, Any
from importlib.resources import files

TableID = Literal['fig2a', 'fig2b', 'fig4a', 'fig4b', 'fig6a', 'fig6b']

SB99_TABLES: dict[TableID, str] = {
    'fig2a': 'fig2a.dat',
    'fig2b': 'fig2b.dat',
    'fig4a': 'fig4a.dat',
    'fig4b': 'fig4b.dat',
    'fig6a': 'fig6a.dat',
    'fig6b': 'fig6b.dat',
}

def _get_path(table_name: TableID) -> Path:
    """
    Internal helper to resolve the full path of a Starburst99 SED table.

    Parameters
    ----------
    table_name : TableID
        Strict table identifier (e.g., 'fig2a').

    Returns
    -------
    Path
        Pathlib object pointing to the file

    Raises
    ------
    ValueError
        If table_name is not one of the available Starburst99 tables.
    """
    if table_name not in SB99_TABLES:
        valid_keys = ', '.join(SB99_TABLES.keys())
        raise ValueError(f'Unknown table "{table_name}". Available tables: {valid_keys}.')

    filename = SB99_TABLES[table_name]
    
    return files(__name__).joinpath(filename)

def load_table(table_name: TableID, **kwargs: Any) -> npt.NDArray[np.float64]:
    """
    Load one of the available Starburst99 (Leitherer et al. 1999) tables into a numpy array.

    Parameters
    ----------
    table_name : Literal['fig2a', 'fig2b', 'fig4a', 'fig4b', 'fig6a', 'fig6b']
        Identifier for the Starburst99 available table.
        Must be one of the specific figure IDs from Leitherer et al. (1999).
    **kwargs : Any
        Additional keyword arguments passed directly to `numpy.loadtxt`

    Returns
    -------
    npt.NDArray[np.float64]
        The data loaded from the file as a standard numpy array.

    Raises
    ------
    FileNotFoundError
        If the data file is missing from the package installation.
    ValueError
        If `table_name` is invalid.
    
    Examples
    --------
    >>> # Load table fig2a skipping first wavelength row
    >>> data = load_table('fig2a', skiprows=1)
    >>> data.shape
    (1221, 37)
    """
    file_path = _get_path(table_name)

    if not file_path.is_file():
        raise FileNotFoundError(f'Data file missing: {file_path}')

    return np.loadtxt(str(file_path), **kwargs)

def get_path(table_name: TableID) -> str:
    """
    Get the absolute path to a data file.

    Parameters
    ----------
    table_name : Literal['fig2a', 'fig2b', 'fig4a', 'fig4b', 'fig6a', 'fig6b']
        Identifier for the Starburst99 available table.
    
    Returns
    -------
    str
        Absolute path to the data file.

    Examples
    --------
    >>> path = get_path('fig4b')
    >>> print(path)
    '/usr/local/lib/python3.11/site-packages/my_package/data/fig4b.dat'
    """
    return str(_get_path(table_name))