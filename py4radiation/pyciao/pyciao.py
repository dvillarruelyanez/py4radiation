#!/usr/bin/env python3

import sys
import time
import logging
import argparse
import multiprocessing

from pathlib import Path
from typing import List, Any

MPI_AVAILABLE = False
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    pass

from parser import Config
from tasks import TaskManager, TaskDef
from runner import execute_task, initialize_worker, TaskResult

def setup_main_logging(outdir: str, prefix: str) -> None:
    """
    Set up the main log file for Rank 0.

    Parameters
    ----------
    outdir : str
        Directory to save the log file.
    prefix : str
        Prefix for the log filename.
    """
    logfile = f'{prefix}.log'
    if outdir:
        logfile = f'{outdir}{logfile}'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logging.info(f'Log file initialized at {logfile}.')

def write_runfile(config: Config) -> None:
    """
    Write the summary .run file.

    Parameters
    ----------
    config : Config
        Configuration object.
    """
    par = config.params

    outdir = str(par.get('outputDir', ''))
    prefix = str(par.get('outputFilePrefix', 'ciaorun'))

    runfile = f'{outdir}{prefix}.run'
    logging.info(f'Writing run file: {runfile}.')

    try:
        with open(runfile, 'w') as f:
            f.write(f'# Run started {time.asctime()}\n#\n')
            f.write(f'# cloudyRunMode = {par.get("cloudyRunMode")}\n')
            f.write(f'# outputFilePrefix = {prefix}\n')
            f.write(f'# outputDir = {outdir}\n#\n')

            f.write('# Commands to be executed everytime:\n')
            for cmd in config.const_cmd:
                f.write(f'# {cmd}\n')

            f.write('#\n# Loop commands and values:\n')
            headers = ['#run']

            for item in config.loops_cmd:
                if item['type'] == 'single':
                    cmd_name = item['command'].replace('\n', ' ')
                    f.write(f'# {cmd_name}: {" ".join(str(val) for val in item["values"])}\n')
                    headers.append(cmd_name)

                elif item['type'] == 'set':
                    # Future implementation.
                    continue

            f.write('#\n')
            f.write('\t'.join(headers) + '\n')

            idx = 1
            hden_vals = config.loops_cmd[0]['values']
            fils_vals = config.loops_cmd[1]['values']

            for i in hden_vals:
                for j in fils_vals:
                    row = [str(idx)]
                    row.extend([str(i)])
                    row.extend([str(j)])
                    f.write('\t'.join(row) + '\n')
                    idx += 1
    except IOError as e:
        logging.error(f'Failed to write runfile {runfile}: {e}.')

def run_mpi(config: Config, tasks: List[TaskDef]) -> List[TaskResult]:
    """
    Execute tasks using MPI.
    """
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()

    config = COMM.bcast(config, root=0)
    tasks  = COMM.bcast(tasks, root=0)

    initialize_worker(config, rank=RANK)
    
    my_tasks = tasks[RANK::SIZE]
    local_results = [execute_task(t) for t in my_tasks]
    
    all_results = COMM.gather(local_results, root=0)

    final_results = []
    if RANK == 0 and all_results:
        for rlist in all_results:
            final_results.extend(rlist)
    
    return final_results

def run_multiprocessing(config: Config, tasks: List[TaskDef], nproc: int) -> List[TaskResult]:
    """
    Execute tasks using multiprocessing.
    """
    logging.info(f'Starting multiprocessing pool with {nproc} processes.')

    with multiprocessing.Pool(processes=nproc, initializer=initialize_worker, initargs=(config,)) as pool:
        results_iter = pool.imap_unordered(execute_task, tasks)

        final_results = []
        for i, res in enumerate(results_iter):
            final_results.append(res)
            if (i + 1) % 10 == 0:
                logging.info(f'Progress: {i + 1}/{len(tasks)} tasks completed.')

    return final_results

def main() -> None:
    """
    Main Execution Routine.
    """
    is_mpi = False
    rank = 0

    if MPI_AVAILABLE:
        if MPI.COMM_WORLD.Get_size() > 1:
            is_mpi = True
            rank = MPI.COMM_WORLD.Get_rank()

    config: Config | None = None
    tasks: List[TaskDef] | None = None

    args = None

    if rank == 0:
        parser = argparse.ArgumentParser(
            description='pyCIAO: Hybrid MPI/multiprocessing implementation of CIAOLoop'
        )

        parser.add_argument('parfile', type=Path, help='Path to parameter file (.par).')
        parser.add_argument('--nproc', type=int, default=1, help='Number of processes for multiprocessing (ignored if using MPI).')
        args = parser.parse_args()

        start_time = time.time()

        try:
            config = Config(args.parfile)
            config.parse()

            par = config.params
            setup_main_logging(par.get('outputDir', ''), par.get('outputFilePrefix', 'ciaorun'))

            mode_str = 'MPI' if is_mpi else 'Multiprocessing'
            logging.info(f'--- pyCIAO {mode_str} Run Started ---')

            task_manager = TaskManager(config)
            tasks = list(task_manager.gen_tasks())

            total_tasks = len(tasks)
            logging.info(f'Generated {total_tasks} Cloudy tasks.')

            if total_tasks == 0:
                logging.warning('No tasks were generated. Exiting...')
                sys.exit(0)

            write_runfile(config)
        except Exception as e:
            logging.critical(f'Rank 0 initialization failed: {e}', exc_info=True)
            sys.exit(1)

    final_results = []

    if is_mpi:
        if rank != 0:
            run_mpi(None, None)
        else:
            final_results = run_mpi(config, tasks)
    else:
        final_results = run_multiprocessing(config, tasks, args.nproc)

    if rank == 0:
        final_results.sort(key=lambda x: x['run_index'])

        success = sum(1 for r in final_results if r['status'] == 'success')
        fail    = sum(1 for r in final_results if r['status'] == 'failed')

        for res in final_results:
            if res['status'] == 'failed':
                logging.error(f"Run {res['run_index']} FAILED: {res.get('error')}")

        end_time = time.time()

        logging.info('--- Run Finished ---')
        logging.info(f'Total tasks: {len(tasks)}')
        logging.info(f'Successful:  {success}')
        logging.info(f'Failed:      {fail}')
        logging.info(f'Total execution time: {end_time - start_time:.2f} seconds')
        
        print('--------------------------------------------------')
        print(f'All Runs Finished at {time.asctime()}')
        print('--------------------------------------------------')


if __name__ == '__main__':
    main()