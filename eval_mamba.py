# code adapted from PerAct: https://github.com/peract/peract

import gc
import logging
import os
import sys

import yaml


import hydra
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from omegaconf import DictConfig, ListConfig, OmegaConf
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from torch.multiprocessing import Manager
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator

from peract.agents import peract_mamba
from helpers import utils


from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from rlbench.action_modes.arm_action_modes import ArmActionMode, assert_action_shape
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy("file_system")

JOINT_THRESH = 0.05


class JointsTrajectoryActionMode(ArmActionMode):
    """A sequence of joint configurations representing a trajectory."""

    def __init__(self, points: int):
        self._points = points

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        assert_action_shape(action, (7 * self._points,))
        if np.all(action == 0):
            raise InvalidActionError("No valid trajectory given.")
        action = self._pre_proc_traj(action)
        path = ArmConfigurationPath(scene.robot.arm, action)
        done = False
        path.visualize()
        while not done:
            done = path.step()
            scene.step()
            if self._callable_each_step is not None:
                # Record observations
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break

        path.clear_visualization()

    def _pre_proc_traj(self, action):
        action = action.reshape(-1, 7)
        new_actions = [action[0]]

        for idx in range(1, len(action) - 2):
            diff = new_actions[-1] - action[idx]
            if np.abs(diff).max() > JOINT_THRESH:
                new_actions.append(action[idx])

        new_actions.append(action[-1])
        return np.stack(new_actions, axis=0).reshape(-1)

    def action_shape(self, scene: Scene) -> tuple:
        return (7 * self._points,)

    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step

    def record_end(self, scene, steps=60, step_scene=True):
        if self._callable_each_step is not None:
            for _ in range(steps):
                scene.step()
                self._callable_each_step(scene.get_observation())


def eval_seed(
    train_cfg, eval_cfg, logdir, cams, env_device, multi_task, seed, env_config
) -> None:
    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    train_cfg.method.update(eval_cfg.method)

    if train_cfg.method.name == "HDP":
        method_path = "cfgs/method/mamba.yaml"
        method_path = os.path.join(os.getcwd().split("output")[0], method_path)
        with open(method_path, "r") as f:
            method_cfg = yaml.safe_load(f)

        method_cfg.update(
            {
                "scene_bounds": eval_cfg.env.scene_bounds,
                "robot_offset": eval_cfg.env.robot_offset,
            }
        )
        train_cfg.method.update(method_cfg)
        agent = peract_mamba.launch_utils.create_agent(train_cfg)

    else:
        raise ValueError("Method %s does not exists." % train_cfg.method.name)

    stat_accum = SimpleAccumulator(eval_video_fps=30)

    weightsdir = os.path.join(logdir, "weights")

    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=train_cfg.framework.training_iterations,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task,
    )

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()

    # evaluate all checkpoints (0, 1000, ...) which don't have results, i.e. validation phase
    if eval_cfg.framework.eval_type == "missing":
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))

        env_data_csv_file = os.path.join(logdir, "eval_data.csv")
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            evaluated_weights = sorted(map(int, list(env_dict["step"].values())))
            weight_folders = [w for w in weight_folders if w not in evaluated_weights]

        print("Missing weights: ", weight_folders)

    # pick the best checkpoint from validation and evaluate, i.e. test phase
    elif eval_cfg.framework.eval_type == "best":
        env_data_csv_file = os.path.join(logdir, "eval_data.csv")
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            existing_weights = list(
                map(int, sorted(os.listdir(os.path.join(logdir, "weights"))))
            )
            task_weights = {}
            for task in tasks:
                weights = list(env_dict["step"].values())

                if len(tasks) > 1:
                    task_score = list(env_dict["eval_envs/return/%s" % task].values())
                else:
                    task_score = list(env_dict["eval_envs/return"].values())

                avail_weights, avail_task_scores = [], []
                for step_idx, step in enumerate(weights):
                    if step in existing_weights:
                        avail_weights.append(step)
                        avail_task_scores.append(task_score[step_idx])

                assert len(avail_weights) == len(avail_task_scores)
                best_weight = avail_weights[
                    np.argwhere(avail_task_scores == np.amax(avail_task_scores))
                    .flatten()
                    .tolist()[-1]
                ]
                task_weights[task] = best_weight

            weight_folders = [task_weights]
            print("Best weights:", weight_folders)
        else:
            raise Exception("No existing eval_data.csv file found in %s" % logdir)

    # evaluate only the last checkpoint
    elif eval_cfg.framework.eval_type == "last":
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))
        weight_folders = [weight_folders[-1]]
        print("Last weight:", weight_folders)

    # evaluate a specific checkpoint
    elif type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)

    else:
        raise Exception("Unknown eval type")

    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info(
            "No weights to evaluate. Results are already available in eval_data.csv"
        )
        sys.exit(0)

    # evaluate several checkpoints in parallel
    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    split_n = utils.split_list(num_weights_to_eval, eval_cfg.framework.eval_envs)
    for split in split_n:
        processes = []
        for e_idx, weight_idx in enumerate(split):
            weight = weight_folders[weight_idx]
            env_runner.start(
                weight,
                save_load_lock,
                writer_lock,
                env_config,
                e_idx % torch.cuda.device_count(),
                eval_cfg.framework.eval_save_metrics,
                eval_cfg.cinematic_recorder,
            )

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name="eval", config_path="cfgs")
def main(eval_cfg: DictConfig) -> None:
    logging.info("\n" + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = eval_cfg.framework.logdir

    train_config_path = os.path.join(logdir, "config.yaml")
    if os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception("Missing seed%d/config.yaml" % start_seed)

    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info("Using env device %s." % str(env_device))

    gripper_mode = Discrete()
    arm_action_mode = JointsTrajectoryActionMode(64)
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]
    eval_cfg.rlbench.cameras = (
        eval_cfg.rlbench.cameras
        if isinstance(eval_cfg.rlbench.cameras, ListConfig)
        else [eval_cfg.rlbench.cameras]
    )
    obs_config = utils.create_obs_config(
        eval_cfg.rlbench.cameras,
        eval_cfg.rlbench.camera_resolution,
        train_cfg.method.name,
    )

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single-task or multi-task
    if len(eval_cfg.rlbench.tasks) > 1:
        tasks = eval_cfg.rlbench.tasks
        multi_task = True

        task_classes = []
        for task in tasks:
            if task not in task_files:
                raise ValueError("Task %s not recognised!." % task)
            task_classes.append(task_file_to_task_class(task))

        env_config = (
            task_classes,
            obs_config,
            action_mode,
            eval_cfg.rlbench.demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            eval_cfg.framework.eval_episodes,
            train_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )
    else:
        task = eval_cfg.rlbench.tasks[0]
        multi_task = False

        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_class = task_file_to_task_class(task)

        env_config = (
            task_class,
            obs_config,
            action_mode,
            eval_cfg.rlbench.demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            train_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )

    logging.info("Evaluating seed %d." % start_seed)
    eval_seed(
        train_cfg,
        eval_cfg,
        logdir,
        eval_cfg.rlbench.cameras,
        env_device,
        multi_task,
        start_seed,
        env_config,
    )


if __name__ == "__main__":
    main()
