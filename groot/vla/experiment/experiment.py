import logging
import os
import time
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from groot.vla.experiment.base import BaseExperiment, BaseTrainer
from groot.vla.utils.action_args_override_utils import apply_action_overrides

logger = logging.getLogger(__name__)


INITIAL_ACTIONS_FILENAME = "initial_actions.npz"


class ForceRestart(ValueError):
    pass


class VLATrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.benchmark_time = kwargs.pop("benchmark_time", False)
        self.step_timer = None
        self.num_trials = kwargs.pop("num_trials", 10)
        self.curr_trial = 0
        self.all_times = []
        self.start_time = time.time()
        self.restart_max_seconds = kwargs.pop("restart_max_seconds", 0)
        print(f"restart_max_seconds: {self.restart_max_seconds}")

        import torch.distributed as dist

        self.rank = dist.get_rank()

        self.micro_global_step = 0

        super().__init__(**kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        self.micro_global_step += 1
        print(f"[RANK {self.rank} HEARTBEAT] Start training micro step {self.micro_global_step}")

        if hasattr(self.model.action_head, "global_step"):
            self.model.action_head.global_step = self.state.global_step

        if self.benchmark_time:
            if self.state.global_step % 100 == 0:
                if self.step_timer is not None:
                    elapsed_time = time.time() - self.step_timer
                    print(f"Time for 100 steps: {elapsed_time:.2f} seconds")
                    self.all_times.append(elapsed_time)
                    self.curr_trial += 1
                self.step_timer = time.time()  # Reset the timer
            if self.curr_trial >= self.num_trials:
                print(
                    f"Average time for 100 steps in {self.num_trials} trials: {np.mean(self.all_times):.2f} seconds"
                )
                print(
                    f"Average time for 100 steps in last {int(self.num_trials / 2)} trials: {np.mean(self.all_times[-int(self.num_trials / 2):]):.2f} seconds"
                )
                exit(0)
        if self.state.global_step % self.state.save_steps == 1:
            # just finished saving a checkpoint
            if self.restart_max_seconds > 0:
                cur_time = time.time()
                if (cur_time - self.start_time) > self.restart_max_seconds:
                    raise ForceRestart(f"Exceeded time limit {self.restart_max_seconds} seconds")
        loss_dict = super().training_step(model, inputs, *args, **kwargs)
        print(f"[RANK {self.rank} HEARTBEAT] End training micro step {self.micro_global_step}")
        return loss_dict


class VLATrainerInferenceBenchmark(VLATrainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        warmup_steps = 100
        measure_steps = 100

        model.eval()
        print("shapes", "ids", inputs["input_ids"].shape, "pixels", inputs["pixel_values"].shape)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                for i in range(warmup_steps):
                    action = model.module.get_action(inputs)
                    action.keys()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                for i in range(measure_steps):
                    action = model.module.get_action(inputs)
                    action.keys()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        time_per_step = elapsed_time / measure_steps
        print("Time per step: {:.2f} ms".format(time_per_step))
        exit()


class VLAExperiment(BaseExperiment):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Dump the initial actions
        if hasattr(self.train_dataset, "get_initial_actions"):
            # We only dump the initial actions for the real robot dataset
            # Sim dataset doesn't have this function
            """
            initial_actions: list[dict[str, dict[str, np.ndarray]]]
            0: (the first dataset)
                trajectory_name:
                action_key:
                    action: np.ndarray
            1: (the second dataset)
                ...
            """
            initial_actions = self.train_dataset.get_initial_actions()
            if len(initial_actions) > 0:
                initial_actions_path = self.exp_cfg_dir / INITIAL_ACTIONS_FILENAME
                np.savez(str(initial_actions_path), initial_actions)
                print("Successfully dumped initial actions")
            else:
                print("No initial actions to dump")


@hydra.main(config_path="../configs", config_name="conf", version_base=None)
def main(cfg):
    # Automatically update action dim and action horizon keys if specified in the config
    cfg = apply_action_overrides(cfg)

    experiment = VLAExperiment(cfg)
    experiment.train()


if __name__ == "__main__":
    main()
