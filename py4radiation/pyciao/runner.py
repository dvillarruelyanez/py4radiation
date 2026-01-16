#!/usr/bin/env python3

import numpy as np

import re
import sys
import time
import logging
import subprocess

from pathlib import Path
from numpy.typing import NDArray
from typing import TypedDict, Any, Final, Optional

from parser import Config
from tasks import TaskDef

class TaskResult(TypedDict):
    """
    Structure for the result returned by a worker task.
    """
    run_index: int
    status: str
    error: Optional[str]

WORKER_CONFIG: Optional[Config] = None

ATOMIC_MASSES: Final[NDArray[np.float64]] = np.array([
    1.00794, 4.002602, 6.941, 9.012182, 10.811,
    12.0107, 14.0067, 15.9994, 18.9984032, 20.1797,
    22.989770, 24.3050, 26.981538, 28.0855, 30.973761,
    32.065, 35.453, 39.948, 39.0983, 40.078,
    44.955910, 47.867, 50.9415, 51.9961, 54.938049,
    55.845, 58.933200, 58.6934, 63.546, 65.409
])

ELEMENT_DATA: Final[dict[str, Any]] = {
    'atomicNumber': {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
        'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
    },
    'atomicName': {
        'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium', 'B': 'Boron',
        'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen', 'F': 'Fluorine', 'Ne': 'Neon',
        'Na': 'Sodium', 'Mg': 'Magnesium', 'Al': 'Aluminium', 'Si': 'Silicon', 'P': 'Phosphorus',
        'S': 'Sulphur', 'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
        'Sc': 'Scandium', 'Ti': 'Titanium', 'V': 'Vanadium', 'Cr': 'Chromium', 'Mn': 'Manganese',
        'Fe': 'Iron', 'Co': 'Cobalt', 'Ni': 'Nickel', 'Cu': 'Copper', 'Zn': 'Zinc'
    }
}

class CloudyExecutionError(RuntimeError):
    """
    Custom exception for failures during Cloudy subprocess execution.
    """
    pass

def setup_worker_logging(rank: int | None = None) -> None:
    """
    Configure logging for a worker process.

    Parameters
    ----------
    rank : int | None, optional
        The MPI rank ID. If provided, it is included in the log format.
    """
    if rank is not None:
        log_fmt = f'[Rank-{rank}] %(levelname)s - %(message)s'
    else:
        log_fmt = f'[Worker-%(process)d] %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

def initialize_worker(config: Config, rank: int | None = None) -> None:
    """
    Initialize the worker process with configuration and logging.

    Parameters
    ----------
    config : Config
        Parsed configuration object.
    rank : int | None, optional
        MPI rank of this worker
    """
    global WORKER_CONFIG
    WORKER_CONFIG = config
    setup_worker_logging(rank)

# ---------------------------------------------------------------------

def execute_task(task: TaskDef) -> TaskResult:
    """
    Execute a single task.

    Parameters
    ----------
    task : TaskDef
        Task dictionary containing run parameters

    Returns
    -------
    TaskResult
        Dictionary summarizing the success or failure of the run.
    """
    global WORKER_CONFIG

    if WORKER_CONFIG is None:
        return {
            'run_index': task.get('run_index', -1),
            'status': 'failed',
            'error': 'Worker not initialized'
        }

    run_index = task['run_index']
    run_mode  = int(task['run_mode'])

    start_time = time.time()
    logging.info(f'Run {run_index} STARTED.')

    try:
        if run_mode == 1:
            heatingcooling(task, WORKER_CONFIG)
        elif run_mode == 3:
            ionfractions(task, WORKER_CONFIG)
        else:
            raise ValueError(f'Unsupported run mode: {run_mode}.')

        elapsed = time.time() - start_time
        logging.info(f'Run {run_index} FINISHED in {elapsed:.2f} s.')

        return {
            'run_index': run_index,
            'status': 'success',
            'error': None
        }

    except Exception as e:
        logging.error(f'Run {run_index} FAILED', exc_info=True)
        return {
            'run_index': run_index,
            'status': 'failed',
            'error': str(e)
        }

def run_subprocess(commands: list[str], infile: Path, outfile: Path, cloudy_exe: str) -> bool:
    """
    Write input commands to file and execute Cloudy via subprocess.

    Parameters
    ----------
    commands : list[str]
        List of Cloudy commands to write to the input file.
    infile : Path
        Path to the input file.
    outfile : Path
        Path to the output file.
    cloudy_exe : str
        Path for the Cloudy executable.

    Returns
    -------
    bool
        True if Cloudy crashed or exited with error, False otherwise.

    Raises
    ------
    IOError
        If file writing fails.
    CloudyExecutionError
        If the subprocess fails to launch.
    """
    # 1. Input file
    try:
        with open(infile, 'w') as f:
            for cmd in commands:
                f.write(f'{cmd}\n')
    except OSError as e:
        raise IOError(f'Failed to write input file {infile}.') from e

    # 2. Execute Cloudy
    try:
        with open(outfile, 'w') as f_out, open(infile, 'r') as f_in:
            result = subprocess.run(
                [cloudy_exe],
                stdin=f_in,
                stdout=f_out,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

    except FileNotFoundError:
        raise FileNotFoundError(f'Cloudy executable not found at: {cloudy_exe}.')
    except subprocess.SubprocessError as e:
        raise CloudyExecutionError(f'Subprocess failed to launch: {e}.') from e
    except OSError as e:
        raise IOError(f'IO error during subprocess execution: {e}.') from e

    # 3. Check crashes/errors
    crashed = True
    try:
        if outfile.exists():
            with open(outfile, 'r') as f:
                lines = f.readlines()
                if lines and any('Cloudy exited OK' in line for line in lines[-10:]):
                    crashed = False
    except OSError:
        pass

    if result.returncode != 0 or crashed:
        logging.warning(f'Cloudy reported an error or crashed for {infile}.')
        if result.stderr:
            logging.warning(f'Cloudy stderr: {result.stderr.strip()}.')
        return True

    return False

# ---------------------------------------------------------------------

def _get_temperature_loop(config: Config) -> list[float]:
    """
    Calculate the temperature points for the inner loop.
    """
    par = config.params
    try:
        tmin = float(par['coolingMapTmin'])
        tmax = float(par['coolingMapTmax'])

        tpts = par.get('coolingMapTpoints')
        tlog = par.get('coolingMapdLogT')

        if tpts and tlog:
            raise ValueError('coolingMapTpoints and coolingMapdLogT cannot both be set')
        
        temperatures: list[float] = []

        if tpts:
            points = int(tpts)
            if points == 1:
                return [tmin]
            logtmin = np.log10(tmin)
            logtmax = np.log10(tmax)
            dlogT = (logtmax - logtmin) / (points - 1)
            for i in range(points):
                temperatures.append(10**(logtmin + i * dlogT))

        elif tlog:
            step = float(tlog)
            current_t = tmin
            while current_t <= tmax * (1 + 1e-9):
                temperatures.append(current_t)
                current_t *= 10**step

        else:
            raise ValueError('Must specify either coolingMapTpoints or coolingMapdLogT')
        
        return temperatures
    
    except Exception as e:
        raise ValueError(f'Error initializing temperature loop: {e}')

def _get_loops(task: TaskDef, config: Config) -> list[str]:
    """
    Extract specific loop command strings used in this task.
    """
    loops = []
    loop_keys = [
        item['command'].split()[0] for item in config.loops_cmd if 'command' in item
    ]

    for cmd in task['base_commands']:
        for key in loop_keys:
            if cmd.startswith(key):
                loops.append(cmd)
                break
    return loops

def _get_physical_conditions(filepath: Path) -> tuple[float, float]:
    """
    Parse H density (hden) and electron density (eden)
    from physical conditions file.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if not lines:
                return 0.0, 0.0

        last_line = lines[-1].strip()
        parts = last_line.split('\t')

        if len(parts) >= 4:
            hden = float(parts[2])
            eden = float(parts[3])
            return hden, eden
        return 0.0, 0.0
    except (ValueError, IndexError, OSError):
        return 0.0, 0.0
    
def _get_cooling(filepath: Path) -> tuple[float, float]:
    """
    Parse heating and cooling rates from cooling file.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in reversed(lines):
            if not line.lstrip().startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 4:
                    return float(parts[2]), float(parts[3])
        raise ValueError('No valid data line found.')
    except Exception:
        return 0.0, 0.0

def _get_mmw(abunfile: Path, ionsfile: Path) -> float:
    """
    Calculate Mean Molecular Weight (MMW).
    """
    rel_abund = []
    try:
        with open(abunfile, 'r') as f:
            lines = [l for l in f if l.strip() and not l.startswith('#')]

            if lines:
                vals = [float(x) for x in lines[-1].split()]
                if vals:
                    hval = vals[0]
                    rel_abund = [10**(x - hval) for x in vals]

    except OSError:
        return 0.0

    if not rel_abund:
        return 0.0

    num_re = re.compile(r'[+-]?\d+\.\d+(?:[Ee][+-]?\d+)?')

    total_mass = 0.0
    total_particles = 0.0
    idx = 0
    procs = False

    try:
        with open(ionsfile, 'r') as f:
            lines = f.readlines()

        if not lines:
            raise IOError(f'Failed to parse ionisation file.')

        for line in lines:
            if 'Hydrogen' in line:
                procs = True
            if not procs:
                continue

            if re.match(r'^\s*[a-zA-Z]', line):
                logfracs = [float(x) for x in num_re.findall(line)]

                if idx < len(ATOMIC_MASSES) and idx < len(rel_abund):
                    z = idx + 1
                    fracs = logfracs[:z+1]
                    
                    abund = rel_abund[idx]
                    mass = ATOMIC_MASSES[idx]
                    total_mass += abund * mass

                    for w, logfrac in enumerate(fracs):
                        linfrac = 10**logfrac
                        total_particles += abund * linfrac * (w + 1)

                    idx += 1
        
        if total_particles == 0.0:
            return 0.0
        
        return total_mass / total_particles

    except Exception:
        return 0.0

def _get_ion_fraction(ionsfile: Path, elements: list[str]) -> dict[str, list[float]]:
    """
    Parse ionisation fractions for specific elements.
    """
    name2symbol = {v: k for k, v in ELEMENT_DATA['atomicName'].items()}
    atomic_numbers = ELEMENT_DATA['atomicNumber']
    elements_data: dict[str, list[float]] = {el: [] for el in elements}

    num_re = re.compile(r'[+-]?\d+\.\d+(?:[Ee][+-]?\d+)?')

    try:
        with open(ionsfile, 'r') as f:
            lines = f.readlines()

        idx = -1
        for i, line in enumerate(lines):
            if 'Hydrogen' in line:
                idx = i

        if idx == -1:
            raise ValueError('No ionisation data block found')
        
        for line in lines[idx:]:
            line = line.strip()
            if not line:
                continue

            match = re.match(r'^\s*([a-zA-Z]+)', line)
            if match:
                element_name = match.group(1)
                symbol = name2symbol.get(element_name)

                if symbol in elements_data:
                    vals = num_re.findall(line)
                    logfracs = [float(x) for x in vals]

                    if symbol in atomic_numbers:
                        z = atomic_numbers[symbol]
                        expected_ions = z + 1

                        if len(logfracs) < expected_ions:
                            padding = [-30.0] * (expected_ions - len(logfracs))
                            logfracs.extend(padding)

                        elements_data[symbol] = logfracs[:expected_ions]

            if symbol not in elements_data:
                continue

        return elements_data
        
    except Exception as e:
        raise IOError(f'Failed to parse ion fractions from {ionsfile}: {e}')

# -----------------------------------------------------------------------------------------
    
def heatingcooling(task: TaskDef, config: Config) -> None:
    """
    Execute Mode 1: Heating and Cooling Maps.
    """
    key_name = task['key_name']
    datfile_path = Path(f'{key_name}.dat')

    temp_suffix  = '.temp'
    temp_infile  = Path(f'{key_name}.cloudyIn{temp_suffix}')
    temp_outfile = Path(f'{key_name}.cloudyOut{temp_suffix}')

    temp_heatfile = Path(f'{key_name}.heating{temp_suffix}')
    temp_coolfile = Path(f'{key_name}.cooling{temp_suffix}')

    temp_abunfile = Path(f'{key_name}.abundance{temp_suffix}')
    temp_ionsfile = Path(f'{key_name}.ionization{temp_suffix}')
    temp_physfile = Path(f'{key_name}.physical{temp_suffix}')

    temp_toclean = [
        temp_infile,
        temp_outfile,
        temp_heatfile,
        temp_coolfile,
        temp_abunfile,
        temp_ionsfile,
        temp_physfile
    ]

    loop_vals = _get_loops(task, config)

    with open(datfile_path, 'w') as f:
        f.write(f'# {time.asctime()}\n#\n')
        f.write(f'# Cooling Map File\n#\n')
        f.write(f'# Loop values:\n')
        for val in loop_vals:
            f.write(f'# {val}\n')
        f.write('#\n')
        f.write('# Data Columns:\n')
        f.write('# 1: Te [K]\n')
        f.write('# 2: Heating [erg s^-1 cm^3]\n')
        f.write('# 3: Cooling [erg s^-1 cm^3]\n')
        f.write('# 4: Mean Molecular Weight [amu]\n#\n')
        f.write('#Te\t\tHeating\t\tCooling\t\tMMW\n')

    temperatures = _get_temperature_loop(config)
    scale_factor = int(config.params.get('coolingScaleFactor', 1))

    for temp in temperatures:
        commands = list(task['base_commands'])

        commands.append(f'constant temperature {temp:.6e} K linear')
        commands.append(f'punch last cooling file = "{temp_coolfile}"')
        commands.append(f'punch last heating file = "{temp_heatfile}"')
        commands.append(f'punch last abundance file = "{temp_abunfile}"')
        commands.append(f'punch last ionization means file = "{temp_ionsfile}"')
        commands.append(f'punch last physical conditions file = "{temp_physfile}"')

        crashed = run_subprocess(commands, temp_infile, temp_outfile, str(config.params['cloudyExe']))

        heating, cooling, mmw = 0.0, 0.0, 0.0

        if crashed:
            logging.warning(f'Run {task["run_index"]}, T={temp:.2e} K: Cloudy crashed. Writing zeros ... ')

        else:
            try:
                hden, eden = _get_physical_conditions(temp_physfile)
                heating, cooling = _get_cooling(temp_coolfile)

                if scale_factor == 1:
                    factor = hden * hden
                elif scale_factor == 2:
                    factor = hden * eden
                else:
                    raise ValueError(f'Invalid coolingScaleFactor: {scale_factor}')
                
                if factor > 0:
                    heating /= factor
                    cooling /= factor

                mmw = _get_mmw(temp_abunfile, temp_ionsfile)

            except Exception as e:
                logging.error(f'Run {task["run_index"]}, T={temp:.2e} K: Post-processing failed: {e}')

        with open(datfile_path, 'a') as f:
            f.write(f'{temp:.6e}\t{heating:.6e}\t{cooling:.6e}\t{mmw:.6f}\n')

    for f in temp_toclean:
        f.unlink(missing_ok=True)

def ionfractions(task: TaskDef, config: Config) -> None:
    """
    Execute Mode 3: Ionisation Fraction Maps.
    """
    key_name = task['key_name']

    elements = str(config.params.get('ionFractionElements', ''))
    elements = elements.split()

    if not elements:
        raise ValueError('ionFractionElements not specified')

    temp_suffix  = '.temp'
    temp_infile  = Path(f'{key_name}.cloudyIn{temp_suffix}')
    temp_outfile = Path(f'{key_name}.cloudyOut{temp_suffix}')
    temp_ionsfile = Path(f'{key_name}.ionization{temp_suffix}')

    temp_toclean = [temp_infile, temp_outfile, temp_ionsfile]

    loop_vals = _get_loops(task, config)

    dat_files = {}
    for el in elements:
        if el not in ELEMENT_DATA['atomicNumber']:
            logging.warning(f'Run {task["run_index"]}: Skipping unknown element "{el}"')
            continue

        datfile_path = Path(f'{key_name}_{el}.dat')
        dat_files[el] = datfile_path

        with open(datfile_path, 'w') as f:
            f.write(f'# {time.asctime()}\n#\n')
            f.write(f'# Element: {ELEMENT_DATA["atomicName"].get(el, el)} Ion Fraction File\n#\n')
            f.write(f'# Loop values:\n')
            for val in loop_vals:
                f.write(f'# {val}\n')
            f.write('#\n')
            f.write('# Data Columns:\n')
            f.write('# log(Te [K])\n')
            f.write('# log(Ion Fractions)\n#\n')

            num_ions = ELEMENT_DATA['atomicNumber'][el] + 1
            ion_cols = [f'{i + 1}' for i in range(num_ions)]
            f.write(f'#Te\t' + '\t'.join(f'{col}' for col in ion_cols) + '\n')

    temperatures = _get_temperature_loop(config)

    for temp in temperatures:
        commands = list(task['base_commands'])

        commands.append(f'constant temperature {temp:.6e} K linear')
        commands.append(f'punch last ionization means file = "{temp_ionsfile}"')

        crashed = run_subprocess(commands, temp_infile, temp_outfile, str(config.params['cloudyExe']))

        all_ion_fractions = {}
        if crashed:
            logging.warning(f'Run {task["run_index"]}, T={temp:.2e} K: Cloudy crashed')
            for el in dat_files:
                num_ions = ELEMENT_DATA['atomicNumber'][el] + 1
                all_ion_fractions[el] = [-30.0] * num_ions
        
        else:
            try:
                all_ion_fractions = _get_ion_fraction(temp_ionsfile, elements)
            except Exception as e:
                logging.error(f'Run {task["run_index"]}, T={temp:.2e} K: Ion fraction parsing failed: {e}')
                for el in dat_files:
                    num_ions = ELEMENT_DATA['atomicNumber'][el] + 1
                    all_ion_fractions[el] = [-30.0] * num_ions

        log_temp = np.log10(temp)
        for el, f_path in dat_files.items():
            fractions = all_ion_fractions.get(el, [])
            with open(f_path, 'a') as f:
                fracs = [f'{frac:.3f}' for frac in fractions]
                f.write(f'{log_temp:.3f}\t' + '\t'.join(fracs) + '\n')

    for f in temp_toclean:
        f.unlink(missing_ok=True)