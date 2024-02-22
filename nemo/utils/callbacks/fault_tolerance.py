# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import itertools
import os
import pathlib
import queue
import random
import signal
import sys
import threading
import time
import warnings
from dataclasses import fields
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

import fault_tolerance as ft
from nemo.utils import logging


class FaultToleranceCallback(Callback):
    """
    FaultToleranceCallback class is a Torch Lightning callback that handles fault tolerance.
    """

    MIN_ITERS_FOR_TIMEOUT_UPDATE = 2

    def __init__(self, autoresume=False, calculate_timeouts=False, simulated_fault_params=None):
        self.fault_tol_client = None
        self.num_iters_total = 0
        self.num_iters_after_save = 0
        self.saved_checkpoint = False
        self.loaded_checkpoint = False
        self.exception = None
        self.autoresume = autoresume
        self.calculate_timeouts = calculate_timeouts
        self.simulated_fault_params = simulated_fault_params
        self._verify_env()

    def _verify_env(self):
        if self.autoresume and not os.environ.get('FAULT_TOL_FINISHED_FLAG_FILE', ''):
            raise RuntimeError(
                "'FAULT_TOL_FINISHED_FLAG_FILE' env variable is not set. " "Was this job launched with FT launcher?"
            )

    def _setup_fault_tolerance(self, trainer, pl_module):

        get_emergency_state_dict_cb, save_emergency_checkpoint_cb = self._get_fault_tol_callbacks(pl_module)

        # FT client gets full config from the server
        self.fault_tol_client = ft.RankMonitorClient()

        self.fault_tol_client.init_workload_monitoring(
            get_state_dict_cb=get_emergency_state_dict_cb, save_checkpoint_cb=save_emergency_checkpoint_cb,
        )

        if self.simulated_fault_params:
            self._setup_simulated_fault()

        ft_timeouts = self.fault_tol_client.timeouts
        if ft_timeouts.are_valid:
            logging.info(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")
        else:
            if self.calculate_timeouts:
                logging.info(f"Fault tolerance doesn't have valid timeouts yet. Need to collect more data.")
            else:
                raise RuntimeError(
                    "Fault tolerance doesn't have valid timeouts set and 'calculate_timeouts' is False."
                )

    def _get_fault_tol_callbacks(self, pl_module):
        get_state_cb = None
        save_cb = None
        # TODO: extract callbacks from pl_module
        return get_state_cb, save_cb

    def setup(self, trainer, pl_module, stage):
        if self.fault_tol_client is None:
            self._setup_fault_tolerance(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            no_error = self.exception is None
            # if exiting AND just 0 or 1 training iterations were made AND error is not set,
            # assume training has finished successfully and there is nothing else to do.
            # 1 iteration is made when we run a workload for which 'max_time' elapsed,
            # so need to handle that special case.
            if self.autoresume and (self.num_iters_total <= 1 and no_error):
                self._create_finished_flag_file()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.loaded_checkpoint = True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.saved_checkpoint = True

    def _maybe_update_timeouts(self):
        # No need to update timeouts if we don't want to calculate them
        if not self.calculate_timeouts:
            return
        # No need to update timeouts if they were already calculated
        # NOTE: we update timeouts that were predefined in the config
        if self.fault_tol_client.timeouts.are_valid and self.fault_tol_client.timeouts.were_calculated:
            return
        # Ensure that we have adequate data to update timeouts:
        # - There was checkpoint loading
        #   (this can increase time needed to get to the first iter)
        # - There were some iters after checkpoint saving
        #   (saving can make time between subsequent iters longer)
        # - We got minimum number of iters (arbitrary number)
        if (
            self.num_iters_total >= FaultToleranceCallback.MIN_ITERS_FOR_TIMEOUT_UPDATE
            and self.num_iters_after_save > 0
            and self.loaded_checkpoint
        ):
            self.fault_tol_client.calculate_and_set_timeouts()
            logging.info(f'Updated FT timeouts. New values: {self.fault_tol_client.timeouts}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.fault_tol_client.send_heartbeat()
        self.num_iters_total += 1
        if self.saved_checkpoint:
            self.num_iters_after_save += 1
        self._maybe_update_timeouts()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.fault_tol_client.send_heartbeat()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.fault_tol_client.send_heartbeat()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.fault_tol_client.send_heartbeat()

    def on_exception(self, trainer, pl_module, exception):
        self.exception = exception

    def _create_finished_flag_file(self):
        try:
            flag_file_path = pathlib.Path(os.environ["FAULT_TOL_FINISHED_FLAG_FILE"])
            flag_file_path.touch()
        except Exception as e:
            print(f"_create_finished_flag_file exception: {e}", file=sys.stderr)

    def _setup_simulated_fault(self):

        # TODO: this if for testing only, should be removed in release version

        rng = random.Random()

        fault_desc = self.simulated_fault_params

        logging.info(f"Initializing simulated fault: {fault_desc}")

        rank = torch.distributed.get_rank()
        rand_rank = rng.randint(0, torch.distributed.get_world_size() - 1)
        rank_to_fail = int(fault_desc.get('rank_to_fail', rand_rank))
        rank_to_fail = torch.tensor([rank_to_fail], device=torch.cuda.current_device())
        torch.distributed.broadcast(rank_to_fail, 0)
        rank_to_fail = int(rank_to_fail.item())

        if rank != rank_to_fail:
            return

        fault_type = fault_desc.fault_type
        if fault_type == 'random':
            fault_type = rng.choice(['rank_killed', 'rank_hang'])

        if fault_type == 'rank_killed':
            target_pid = os.getpid()
        elif fault_type == 'rank_hang':
            target_pid = os.getpid()
        else:
            raise Exception(f"Unknown fault type {fault_type}")

        delay = fault_desc.base_delay + fault_desc.get('rand_delay', 0) * random.random()

        print(f"Selected fault={fault_type}; target rank={rank_to_fail}; delay={delay}", file=sys.stderr)

        def __fault_thread():
            time.sleep(delay)
            print(f"\n####\nSimulating fault: {fault_type}; rank to fail: {rank_to_fail}\n#####\n", file=sys.stderr)
            if fault_type == 'rank_hang':
                os.kill(target_pid, signal.SIGTSTP)
            else:
                os.kill(target_pid, signal.SIGKILL)

        fault_sim_thread = threading.Thread(target=__fault_thread)
        fault_sim_thread.daemon = True
        fault_sim_thread.start()
