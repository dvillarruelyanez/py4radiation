#!/usr/bin/env python3

from pathlib import Path
from typing import Literal

ResolutionType = Literal['LOW', 'HIGH']

class ParameterFiles:
    """
    Generates parameter (.par) files for CIAOLoop.

    Handles configuration for Ion Fraction maps (ib) and
    Heating & Cooling maps (hc)
    """

    def __init__(
        self,
        cloudy_path: str | Path,
        run_name: str,
        elements: str,
        redshift: str = '0.0000e+00',
        resolution: ResolutionType = 'LOW'
    ) -> None:
        """
        Initialise the parameter generator.

        Parameters
        ----------
        cloudy_path : str | Path
            Path to the Cloudy executable.
        run_name : str
            Name for the current CIAOLoop run.
        elements : str
            String of elements for ion fraction calculation (e.g., 'H C N O').
        resolution : {'LOW', 'HIGH'}, optional
            LOW: 81 log T points, 27 log hden points (coarse grid).
            HIGH: 321 log T points, 105 log hden points (fine grid).
        """
        self.cloudy_path = str(cloudy_path)
        self.run_name = run_name
        self.elements = elements
        self.redshift = redshift

        self.root_dir = Path.cwd()

        match resolution:
            case 'LOW':
                self.tpoints = 81
                self.hden_step = 0.5
            case 'HIGH':
                self.tpoints = 321
                self.hden_step = 0.125
            case _:
                raise ValueError(f'Resolution must be "LOW" or "HIGH". Got {resolution}.')

    def _get_header(self, output_subdir: str, run_mode: Literal[1, 3]) -> list[str]:
        """
        Generate the header lines shared by both parameter files.

        Parameters
        ----------
        output_subdit : str
            Output directory for selected mode.
        run_mode : {1, 3}
            Run (1) Heating & Cooling map or (3) Ion Fraction map.

        Returns
        -------
        list[str]
            Header of parameter file.
        """
        outdir_path = self.root_dir / output_subdir

        return [
            "#####################################################",
            "################## RUN  PARAMETERS ##################",
            "",
            "# path to CLOUDY executable",
            f"cloudyExe              = {self.cloudy_path}",
            "",
            "# save raw output from CLOUDY",
            "saveCloudyOutputFiles   = 0",
            "",
            "# exit if CLOUDY crashes",
            "exitOnCrash             = 1",
            "",
            "# run name",
            f"outputFilePrefix       = {self.run_name}",
            "",
            "# output path",
            f"outputDir              = {outdir_path}",
            "",
            "# index of first run",
            "runStartIndex           = 1",
            "",
            "# TEST",
            "test                    = 0",
            "",
            "# run mode",
            f"cloudyRunMode          = {run_mode}",
            ""
        ]

    def _get_loops(self, is_ib: bool) -> list[str]:
        """
        Generate the loop commands section.

        Parameters
        ----------
        is_ib : bool
            Add a parameter for Ion Fraction maps.
        """
        sed_pattern = self.root_dir / f"{self.run_name}_z*.out"

        if is_ib:
            init_cmd = f'loop [init "{sed_pattern}"] {self.redshift} 0.0001e+00'
        else:
            init_cmd = f'loop [init "{sed_pattern}"] {self.redshift}'

        return [
            "#####################################################",
            "####################### LOOPS #######################",
            "",
            "command stop zone 1",
            "",
            "command iterate to convergence",
            "",
            f"loop [hden] (-9;4;{self.hden_step})",
            "",
            init_cmd
        ]

    def hc_parfile(self) -> Path:
        """
        Generate and write the CIAOLoop parameter file for heating & cooling.
        """
        filename = f'{self.run_name}_hc.par'
        file_path = self.root_dir / filename

        lines = self._get_header(output_subdir="hc", run_mode=1)

        lines.extend([
            "#####################################################",
            "########## HEATING & COOLING MAP PARAMETERS #########",
            "",
            "# min T",
            "coolingMapTmin = 1e1",
            "",
            "# max T",
            "coolingMapTmax = 1e9",
            "",
            "# T resolution (log points)",
            f"coolingMapTpoints = {self.tpoints}",
            "",
            "# scale factor (1 - n_H^2, 2 - n_H * n_e)",
            "coolingScaleFactor = 1",
            ""
        ])

        lines.extend(self._get_loops(is_ib=False))

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')
            
        return file_path

    def ib_parfile(self) -> Path:
        """
        Generate and write the CIAOLoop parameter file for ion fractions.
        """
        filename = f'{self.run_name}_ib.par'
        file_path = self.root_dir / filename

        lines = self._get_header(output_subdir='ib', run_mode=3)

        lines.extend([
            "#####################################################",
            "############ ION FRACTION MAP PARAMETERS ############",
            "",
            "# min T",
            "coolingMapTmin = 1e1",
            "",
            "# max T",
            "coolingMapTmax = 1e9",
            "",
            "# T resolution (log points)",
            f"coolingMapTpoints = {self.tpoints}",
            "",
            "# elements",
            f"ionFractionElements = {self.elements}",
            ""
        ])

        lines.extend(self._get_loops(is_ib=True))

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')

        return file_path