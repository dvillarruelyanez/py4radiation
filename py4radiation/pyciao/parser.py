#!/usr/bin/env python3

import numpy as np

import re
import logging
from pathlib import Path
from typing import Any, Final
from numpy.typing import NDArray

RE_FOR: Final[re.Pattern] = re.compile(r'\((-?\d*\.?\d*)\;(-?\d*\.?\d*)\;(-?\d*\.?\d*)\)')
RE_CMD: Final[re.Pattern] = re.compile(r'\[(.+?)\](.+)')
RE_SET: Final[re.Pattern] = re.compile(r'loop\s*\{')
RE_PAR: Final[re.Pattern] = re.compile(r'^([a-zA-Z0-9_]+)\s*[=\s]\s*(.+)')

class Config:
    """
    Parses the parameter file to configure pyCIAOLoop and Cloudy execution.

    Attributes
    ----------
    parfile : Path
        Path to the parameter file.
    params : dict[str, Any]
        General configuration parameters.
    const_cmd : list[str]
        Cloudy commands to be executed in every run.
    loops_cmd : list[str]
        Loop parameters and values
    modes : dict[int, str]
        Supported run modes.
    """

    def __init__(self, parfile: str | Path) -> None:
        """
        Initialize the Config parser.

        Parameters
        ----------
        parfile : str | Path
            Path to the input parameter file (.par).
        
        Raises
        ------
        FileNotFoundError
            If the provided parameter file does not exist.
        """
        self.parfile = Path(parfile)

        if not self.parfile.exists():
            raise FileNotFoundError(f'Parameter file not found: {self.parfile}.')

        self.params: dict[str, Any] = {}
        self.const_cmd: list[str] = []
        self.loops_cmd: list[dict[str, Any]] = []

        self.modes: dict[int, str] = {
            1: 'HeatingCoolingMap',
            3: 'IonFractionMap'
        }

    def _parse_loops(self, val: str) -> NDArray[np.float64]:
        """
        Parse a loop value string into a range or a list of values.

        Handles 'for-loop' syntax (start;end;step) or standard space/comma
        separated lists.

        Parameters
        ----------
        val : str
            Value containing loop definitions.

        Returns
        -------
        NDArray[np.float64] | list[str]
            Numpy array of values if range is detected, otherwise list of
            string values.

        Raises
        ------
        ValueError
            If the loop syntax is invalid or results in an infinite loop.
        """
        val = val.strip()

        match = RE_FOR.search(val)
        if match:
            try:
                start = float(match.group(1))
                end   = float(match.group(2))
                step  = float(match.group(3))
            except ValueError as err:
                raise ValueError(f'Invalid for loop parameter: {val}.') from err

            if step == 0:
                raise ValueError(f'Invalid step (0) in loop: {val}.')

            if (end - start) * step < 0 and step > 0:
                raise ValueError(f'Infinite loop detected: {val}.')
            if (end - start) * step > 0 and step < 0:
                raise ValueError(f'Infinite loop detected: {val}.')

            return np.arange(start, end + step, step)

        return val.replace(',', ' ').split()

    def _validate_config(self) -> None:
        """
        Validate parsed configuration parameters.

        Ensures required keys exist and values are within expected ranges.
        Modifies `self.params` in place to set defaults or fix paths.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.
        """
        try:
            mode = int(self.params['cloudyRunMode'])
            if mode not in self.modes:
                raise ValueError(
                    f'Unsupported cloudyRunMode: {mode}. '
                    f'Supported modes: {self.modes}.'
                )

        except (ValueError, KeyError) as err:
            raise ValueError('Invalid or missing cloudyRunMode. Must be an integer.') from err

        if 'cloudyExe' not in self.params:
            raise ValueError('cloudyExe must be specified in the parameter file.')

        outdir = self.params.get('outputDir', '.')
        if outdir == '.':
            logging.warning('Output directory not set, defaulting to current directory.')
            self.params['outputDir'] = ''
        else:
            path_out = Path(outdir)
            path_out.mkdir(parents=True, exist_ok=True)
            self.params['outputDir'] = f'{path_out}/'

        if 'outputFilePrefix' not in self.params:
            logging.warning('outputFilePrefix not set, defaulting to "ciaorun".')
            self.params['outputFilePrefix'] = 'ciaorun'

        mode_int = int(self.params['cloudyRunMode'])

        if mode_int == 1:
            if not all(k in self.params for k in ['coolingMapTmin', 'coolingMapTmax']):
                raise ValueError('Heating/Cooling rates requires "coolingMapTmin" and "coolingMapTmax".')

        if mode_int == 3:
            if not all(k in self.params for k in ['coolingMapTmin', 'coolingMapTmax']):
                raise ValueError('Ion Fraction maps requires "coolingMapTmin" and "coolingMapTmax".')

            if 'ionFractionElements' not in self.params:
                logging.warning('ionFractionElements not set, defaulting to "H C N O".')
                self.params['ionFractionElements'] = 'H C N O'

    def parse(self) -> None:
        """
        Read and parse the parameter file.

        Raises
        ------
        ValueError
            If line syntax is incorrect.
        IOError
            If file reading fails.
        """
        logging.info(f'Parsing parameter file: {self.parfile}.')

        try:
            with open(self.parfile, 'r') as f:
                lines = f.readlines()

        except IOError as e:
            raise IOError(f'Failed to read parameter file {self.parfile}: {e}.')

        for n, line in enumerate(lines, 1):
            rline = line.strip().split('#', 1)[0].strip()

            if not rline:
                continue

            low = rline.lower()

            try:
                if low.startswith('command'):
                    cmd = rline.split(maxsplit=1)
                    if len(cmd) > 1:
                        self.const_cmd.append(cmd[1])
                    else:
                        logging.warning(f'Line {n}: Empty command ignored.')

                elif low.startswith('loop'):
                    match = RE_CMD.search(rline)
                    if not match:
                        raise ValueError(f'Invalid loop syntax: {line}.')

                    cmd = match.group(1).strip()
                    val = match.group(2).strip()
                    val = self._parse_loops(val)

                    self.loops_cmd.append({
                        'type': 'single',
                        'command': cmd,
                        'values': val
                    })
                else:
                    match = RE_PAR.match(rline)
                    if match:
                        key = match.group(1).strip()
                        val = match.group(2).strip()
                        self.params[key] = val
                    else:
                        logging.debug(f"Line {n}: Unrecognized format, skipping: {line_content}")
                        continue
            except Exception as e:
                logging.error(f'Error parsing line {n}: "{rline}" -> {e}.')

        self._validate_config()
        logging.info('Configuration parsed succesfully.')