#!/usr/bin/env python3

import numpy as np

import logging
import itertools
from numpy.typing import NDArray
from typing import Iterator, Any, TypedDict, cast

from parser import Config

class TaskDef(TypedDict):
    run_index: int
    key_name: str
    base_commands: list[str]
    run_mode: int

class TaskManager:
    """
    Generates execution tasks based on the parsed configuration.

    This class handles the combinatorial logic of creating unique Cloudy
    run configurations by iterating over the specified loops and combining
    them with constant commands.

    Attributes
    ----------
    config : Config
        Configuration object containing parsed parameters and commands.
    start_idx : int
        The starting index for run numbering. Default is 1.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the TaskManager.

        Parameters
        ----------
        config : Config
            A valid, parsed Config object.
        """
        self.config = config
        self.start_idx = int(config.params.get('runStartIndex', 1))

    def _create_task(self, run_idx: int, loops_cmd: list[str]) -> TaskDef:
        """
        Create a single task dictionary for a worker to execute.

        Parameters
        ----------
        run_idx : int
            Unique identifier for this run.
        loops_cmd : list[str]
            Specific list of loop commands for this run.

        Returns
        -------
        TaskDef
            Dictionary containing all necessary information to execute the task.
        """
        outdir = self.config.params.get('outputDir', '')
        prefix = self.config.params.get('outputFilePrefix', 'ciaorun')

        key_name = f'{outdir}{prefix}_run{run_idx}'

        final_cmd = list(self.config.const_cmd)
        final_cmd.extend(loops_cmd)

        run: TaskDef = {
            'run_index': run_idx,
            'key_name': key_name,
            'base_commands': final_cmd,
            'run_mode': int(self.config.params['cloudyRunMode'])
        }

        return run

    def _format_cmd(self, template: str, value: float | str | NDArray[np.float64]) -> str:
        """
        Format a Cloudy command string with a specific value.

        If the template contains '*', replace it with the value.
        Otherwise, append the value to the template.

        Parameters
        ----------
        template : str
            Command template.
        value : float | str | NDArray[np.float64]
            Value to insert into the command.

        Returns
        -------
        str
            Formatted command string.
        """
        val = str(value)
        if '*' in template:
            return template.replace('*', val)
        else:
            return f'{template} {val}'

    def gen_tasks(self) -> Iterator[TaskDef]:
        """
        Generate a stream of tasks by iterating over all configured loops.

        Yields
        ------
        Iterator[TaskDef]
            An iterator yielding task dictionaries.

        Examples
        --------
        >>> manager = TaskManager(config)
        >>> for task in manager.gen_tasks():
        ...     print(task['run_index])
        """
        loop_iter = []

        for loop_item in self.config.loops_cmd:
            if loop_item['type'] == 'single':
                cmd = str(loop_item['command'])
                raw_vals = loop_item['values']

                iterable = []
                if isinstance(raw_vals, np.ndarray):
                    vals_arr = cast(NDArray[np.float64], raw_vals)
                    iterable = [(self._format_cmd(cmd, val),) for val in vals_arr]
                else:
                    iterable = [(self._format_cmd(cmd, val),) for val in raw_vals]

                loop_iter.append(iterable)

            elif loop_item['type'] == 'set':
                # Future implementation.
                continue

        if not loop_iter:
            logging.info('No loop commands found. Generating a single task.')
            yield self._create_task(self.start_idx, [])
            return

        run_idx = self.start_idx

        for task_cmd in itertools.product(*loop_iter):
            loops_cmd = [item[0] for item in task_cmd]

            yield self._create_task(run_idx, loops_cmd)
            run_idx += 1