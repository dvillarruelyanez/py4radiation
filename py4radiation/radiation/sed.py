#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import TextIO

from py4radiation import data

# --- Physics Constants ---
SPEED_OF_LIGHT = 2.99792458e10   # cm/s
PLANCK_EV      = 4.135667696e-15 # eV s
RYDBERG_EV     = 13.605693       # eV (1 Ryd)
PARSEC_TO_CM   = 3.08567758e18   # cm
ANGSTROM_TO_CM = 1.0e-8          # cm

class SED:
    """
    A class to load, process and format SEDs for astrophysical codes.

    This class handles the conversion of Leitherer et al. (1999) style
    SED tables into Cloudy-readable formats.

    Attributes
    ----------
    run_name : str
        Name of the run, used for output filenames.
    distance_cm : float
        Luminosity distance to the source in cm.
    data_table : npt.NDArray[np.float64]
        Raw loaded SED data table.
    redshift : float
        Redshift (z) of the source.
        For this version, only consider near Universe (z=0.0000e+00).
    """

    def __init__(
        self,
        run_name: str,
        custom_table: bool = False,
        table_id: data.TableID = 'fig2a',
        table: str | Path = None,
        distance_kpc: float = 1,
        age_myr: int = 3 ) -> None:
        """
        Initialise the SED processor.

        Parameters
        ----------
        run_name : str
            A unique identifier for this run.
        custom_table : bool, optional
            If a custom table (outside the Sb99 tables) is used.
            Default is False.
        table_id : data.TableID, optional
            The identifier of the bundled SED table to load.
            Default is 'fig2a'.
        table : str | Path
            The path to a custom table with only two columns:
            [wavelength (in Angstroms)] [Log10 Luminosity (in erg/s/Angstrom)].
        distance_kpc : float
            Distance to the radiation source in kiloparsecs (kpc).
        age_myr : int, optional
            Age of the starburst in Myr. Used to select the specific column
            in the time-evolution SB99 SED table. Default is 3.
        """
        self.run_name = run_name
        self.redshift = '0.0000e+00'
        
        self.distance_cm = distance_kpc * 1000 * PARSEC_TO_CM

        if custom_table:
            self.data_table = np.loadtxt(str(table))
            self._age_column_idx = 1
        else:
            self.data_table = data.load_table(table_id)
            self._age_column_idx = self._map_age(age_myr)
        
        self.age_myr = age_myr

    def _map_age(self, age: int) -> int:
        """
        Map the starburst age (Myr) to the specific column index.

        Logic follows the specific format of Leitherer et al. tables.
        """
        if 30 <= age <= 100:
            idx = (age // 10) + 18
        elif 200 <= age <= 900:
            idx = (age // 100) + 27
        else:
            idx = age

        if idx >= self.data_table.shape[1]:
            raise ValueError(
                f'Calculated column index {idx} for age {age} Myr '
                f'exceeds table dimensions ({self.data_table.shape[1]} columns).'
            )

        return int(idx)

    def get_sed(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calculates Energy (Ryd) and Intensity (J_nu).

        Returns
        -------
        energy_ryd : np.ndarray
            Photon energies in Rydbergs.
        F_nu : np.ndarray
            Flux F_nu in erg s^-1 cm^-2 Hz^-1
        """

        wavelength_angstrom = self.data_table[:, 0]
        wavelength_cm = wavelength_angstrom * ANGSTROM_TO_CM

        frequency = SPEED_OF_LIGHT / wavelength_cm
        energy_ryd = (frequency * PLANCK_EV) / RYDBERG_EV

        log_luminosity = self.data_table[:, self._age_column_idx]
        luminosity_lambda = 10**log_luminosity

        c_angstrom_s = SPEED_OF_LIGHT * 1e8
        luminosity_nu = luminosity_lambda * (wavelength_angstrom**2) / c_angstrom_s
        f_nu = luminosity_nu / (4 * np.pi * self.distance_cm**2)

        return energy_ryd, f_nu

    def write_outfile(self, output_dir: str | Path = '.') -> Path:
        """
        Write a Cloudy-readable input file.

        Parameters
        ----------
        output_dir : str or Path
            Directory where the output file will be saved.

        Returns
        -------
        Path
            The path to the generated file.

        Raises
        ------
        ValueError
            If SED coverage does not include 1 Ryd.
        """
        energy, f_nu = self.get_sed()
        log_f_nu = np.log10(f_nu)

        if energy[0] > energy[-1]:
            energy = energy[::-1]
            log_f_nu = log_f_nu[::-1]

        idx_1ryd = np.argmin(np.abs(energy - 1.0))

        if not (0.99 <= energy[idx_1ryd] <= 1.01):
            raise ValueError(f'SED coverage does not include 1 Ryd (closest: {energy[idx_1ryd]:.4f}).')

        norm_val = log_f_nu[idx_1ryd]

        filename = f'{self.run_name}_z{self.redshift}.out'
        outpath = Path(output_dir) / filename

        lines = [
            f'# SED profile at {self.age_myr} Myr',
            f'# z = {self.redshift}',
            f'# E [Ryd] log10 (F_nu)'
        ]

        for i in range(len(energy)):
            command = 'interpolate' if i == 0 else 'continue'
            lines.append(f'{command} ({energy[i]:.10f} {log_f_nu[i]:.10f})')

        lines.append(f'f(nu) = {norm_val:.14f} at {energy[idx_1ryd]:.10f} Ryd')
        lines.append('')

        with open(outpath, 'w') as f:
            f.write('\n'.join(lines))

        return outpath
        
