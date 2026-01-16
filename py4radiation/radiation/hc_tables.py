#!/usr/bin/env python3

import numpy as np

import logging
from pathlib import Path
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

class HeatingCoolingTable:
    """
    A class to process and format heating and cooling tables for HD/MHD codes.

    This class reads the outputs of pyCIAO to generate a single table
    containing temperature, mean molecular weight, heating, and cooling.

    Attributes
    ----------
    run_dir : Path
        Directory containing the pyCIAO run file and data maps.
    run_name : str
        The common identifier (stem) of the pyCIAO run.
    """

    def __init__(self, run_dir: str | Path, run_name: str) -> None:
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
        """
        self.run_dir  = Path(run_dir)
        self.run_name = run_name

    def _process_map(self, map_path: Path, hden_log: float) -> NDArray[np.float64]:
        """
        Helper to load a single map and format it as a (N, 4) array.
        """
        raw_data = np.loadtxt(map_path, comments='#', ndmin=2)

        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError(f"Data file {map_path} must have at least 3 columns "
                             f"(temperature, heating, cooling). Got shape {raw.shape}.")

        temp_vals = raw_data[:, 0].astype(float)
        heat_vals = raw_data[:, 1].astype(float)
        cool_vals = raw_data[:, 2].astype(float)

        hden_val = 10.0 ** float(hden_log)
        hden_vals = np.full(temp_vals.shape, hden_val, dtype=float)

        return np.column_stack((hden_vals, temp_vals, heat_vals, cool_vals))

    def write_table(self, outdir: str | Path = '.', outfile: str | None = None) -> Path:
        """
        Write the heating/cooling rates table.

        Parameters
        ----------
        outdir : str or Path, optional
            Directory where the table will be saved.
            Defaults to '.' (current directory).
        outfile : str, optional
            Name of the output file.
            If None, defaults to '{run_name}_cooltable.dat'.
            This file is saved in the current directory.

        Returns
        -------
        Path
            The full path to the generated table.
        """
        clean_run = self.run_name.removesuffix('.run')
        runfile = self.run_dir / f'{clean_run}.run'

        outdir = Path(outdir)
        filename = outfile or f'{clean_run}_cooltable.dat'
        outpath = outdir / filename

        if not runfile.exists():
            raise FileNotFoundError(f'Master run file not found: {runfile}')

        if not outdir.exists():
            outdir.mkdir(parents=True, exist_ok=True)

        print(f'Generating heating/cooling table from {runfile.name} to {outpath}...')

        with open(runfile, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]

        try:
            run_idx = next(i for i, line in enumerate(lines) if line.startswith('#run'))
        except StopIteration:
            raise ValueError(f'Invalid format: Missing "#run" marker in {runfile}.')

        n_runs = len(lines) - run_idx - 1
        hden_grid = np.linspace(-9, 4, n_runs)

        header = "HDEN[cm^-3]  TEMPERATURE[K]  HEATING[erg_cm^3_s^-1]  COOLING[erg_cm^3_s^-1]"

        with open(outpath, 'w', encoding='utf-8') as f_out:
            f_out.write(f'{header}\n')

            for i in range(n_runs):
                map_path = self.run_dir / f'{clean_run}_run{i + 1}.dat'

                if not map_path.exists():
                    raise FileNotFoundError(f'Missing data map: {map_path}')

                data = self._process_map(map_path, hden_grid[i])

                np.savetxt(f_out, data, fmt=['%.7E', '%.7E', '%.7E', '%.7E'], delimiter='  ')

        return outpath