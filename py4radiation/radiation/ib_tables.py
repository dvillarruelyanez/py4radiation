#!/usr/bin/env python3

import h5py
import logging
import numpy as np

from pathlib import Path
from typing import Iterable
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

class IonTables:
    """
    A class to generate ion fraction tables from pyCIAO output files.

    This class parses a simulation run configuration, loads distributed
    ion fraction maps, and consolidates them into a single HDF5 file
    suitable for astrophysical HD/MHD codes.

    Attributes
    ----------
    run_dir : Path
        Directory containing the pyCIAO run file and data maps.
    run_name : str
        The common identifier (stem) of the pyCIAO run.
    """

    def __init__(self, run_dir: str | Path, run_name: str, elements: Iterable[str]) -> None:
        """
        Initialise the processor with the location and ID of the run.

        Parameters
        ----------
        run_dir : str or Path
            The folder containing the pyCIAO run files.
        run_name : str
            The unique identifier for the run.
            The code expects to find '{run_name}.run' and associated
            '{run_name}_runX.dat' files in the run_dir.
        elements : Iterable[str]
            List of elements for the ion fraction tables.
        """
        self.run_dir  = Path(run_dir)
        self.elements = elements

        clean_run = run_name.removesuffix('.run')
        self.runfile = Path(run_dir) / f'{clean_run}.run'
        self.outfile = Path(run_dir) / f'{clean_run}_iontable.h5'

        if not self.runfile.exists():
            raise FileNotFoundError(f'Master run file not found: {runfile}')

    def _parse_runfile(self) -> tuple[list[list[float]], list[str], int]:
        """
        Parse the run file for loop parameters and run count.
        """
        with open(self.runfile, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f]

        parameter_values: list[list[float]] = []
        parameter_names: list[str] = []
        collecting = False
        n_runs = 0

        for i, line in enumerate(lines):
            if collecting:
                if line == '#':
                    collecting = False
                else:
                    key, vals = line.split(':', 1)
                    parameter_names.append(key[2:])
                    parameter_values.append([float(v) for v in vals.split()])
            elif line.startswith('# Loop commands and values'):
                collecting = True
            elif line.startswith('#run'):
                n_runs = len(lines) - i - 1
                break

        grid_shape = [len(v) for v in parameter_values]
        if grid_shape and n_runs != int(np.prod(grid_shape)):
            raise ValueError('Run count does not match parameter grid size.')

        return parameter_values, parameter_names, n_runs

    def _read_map(self, map_path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Read a single '*_runX_ELEMENT.dat' file.
        """
        try:
            data = np.loadtxt(map_path, comments='#', ndmin=2)
        except Exception as e:
            raise RuntimeError(f'Failed to read map file {map_path}: {e}.')

        if data.shape[1] < 2:
            raise ValueError(f'Invalid map format in {map_path}.')

        temperature  = data[:, 0].astype(np.float64)
        ion_fraction = data[:, 1:].astype(np.float64)

        return temperature, ion_fraction      

    def _process_element(self, element: str) -> None:
        """
        Process each element provided in the run.
        """
        logger.info(f'Processing element {element}...')

        parameter_values, parameter_names, n_runs = self._parse_runfile()
        grid_shape = [len(v) for v in parameter_values]

        temperature: NDArray[np.float64] | None = None
        ion_grid: NDArray[np.float64] | None = None

        for run_idx in range(n_runs):
            map_path = self.run_dir / f'{self.run_name}_run{run_idx + 1}_{element}.dat'
            if not map_path.exists():
                raise FileNotFoundError(f'Missing map file: {map_path}.')

            temp, ion_frac = self._read_map(map_path)

            if ion_grid is None:
                temperature = temp
                shape = tuple(grid_shape) + ion_frac.shape
                ion_grid = np.zeros(shape, dtype=np.float64)

            idx = np.unravel_index(run_idx, grid_shape) if grid_shape else (run_idx,)
            ion_grid[idx] = ion_frac

        assert temperature is not None and ion_grid is not None

        ion_grid = np.rollaxis(ion_grid, -1)

        with h5py.File(self.outfile, 'a') as f_out:
            ds = f_out.create_dataset(element, data=ion_grid)
            ds.attrs['Temperature'] = temperature

            for i, values in enumerate(parameter_values, start=1):
                ds.attrs[f'Parameter{i}'] = np.asarray(values, dtype=np.float64)

    def write_table(self) -> Path:
        """
        Write the ion fractions table.
        """
        if not self.elements:
            logger.warning('No elements specified!')
            return

        for element in self.elements:
            self._process_element(element)