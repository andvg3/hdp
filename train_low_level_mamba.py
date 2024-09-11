"""Main script."""

import os

import hydra
import torch

from mamba.dataset.rl_bench_dataset import RLBenchDataset
from mamba.models.network import SimpleModel
from mamba.models.multi_level_mamba import MultiLevelMamba
from mamba.robot import MambaRobot
from mamba.trainer import Trainer
from mamba.utils import load_checkpoint

VALID_MAMBA_VARS = ["gripper_poses", "joint_positions", "multi"]


def _create_agent_fn(
    cfgs,
    device,
    mamba_var="multi",
    sim=True,
    mamba_optim=True,
    mamba_optim_steps=100,
    mamba_lr=10,
    pose_augment=False,
):
    robot = None
    file_path = os.path.join(*__file__.split("/")[:-1])
    if mamba_var in ["joint_positions", "multi"]:
        robot = MambaRobot(
            os.path.join("/", file_path, "panda_urdf/panda.urdf"),
        )
    if robot is not None:
        robot.to(device)

    assert mamba_var in VALID_MAMBA_VARS
    if mamba_var == "multi":
        mamba_pose = hydra.utils.instantiate(
            cfgs,
            mamba_var="gripper_poses",
        )
        mamba_joints = hydra.utils.instantiate(
            cfgs,
            mamba_var="joint_positions",
        )

        mamba_model = MultiLevelMamba(
            {
                "gripper_poses": mamba_pose,
                "joint_positions": mamba_joints,
            },
            mamba_optim=mamba_optim,
            mamba_optim_steps=mamba_optim_steps,
            mamba_lr=mamba_lr,
            pose_augment=pose_augment,
            sim=sim,
        )
    else:
        mamba_model = hydra.utils.instantiate(
            cfgs,
            mamba_var=mamba_var,
        )

    mamba_model.to(device)
    return robot, mamba_model


@hydra.main(
    config_path="cfgs",
    config_name="mamba_config",
    version_base=None,
)
def main(cfgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot, mamba_model = _create_agent_fn(
        cfgs.method,
        device,
        mamba_var=cfgs.mamba_var,
        sim=cfgs.env.name == "sim",
        mamba_optim=cfgs.mamba_optim,
        mamba_optim_steps=cfgs.mamba_optim_steps,
        mamba_lr=cfgs.mamba_lr,
        pose_augment=False,
    )

    if os.path.exists(cfgs.load_model_path):
        load_checkpoint(mamba_model, cfgs.load_model_path, cfgs.method.backbone)

    mamba_model = mamba_model.to(device)

    assert os.path.isdir(cfgs.env.data_path)
    if cfgs.env.name == "sim":
        dataset = RLBenchDataset(
            cfgs.env.tasks,
            cfgs.env.tasks_ratio,
            cfgs.env.cameras,
            cfgs.env.num_episodes,
            data_raw_path=os.path.join(cfgs.env.data_path, "train"),
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            diffusion_var=cfgs.mamba_var,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            use_cached=cfgs.use_cached,
            ds_img_size=cfgs.ds_img_size,
        )

        eval_dataset = RLBenchDataset(
            cfgs.env.tasks,
            cfgs.env.tasks_ratio,
            cfgs.env.cameras,
            cfgs.env.num_episodes // 2,
            data_raw_path=os.path.join(cfgs.env.data_path, "eval"),
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            diffusion_var=cfgs.mamba_var,
            training=False,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            use_cached=cfgs.use_cached,
            ds_img_size=cfgs.ds_img_size,
        )
    else:
        # RLBench complains when importing cv2 before it so we move it here
        from mamba.dataset.realworld_dataset import RealWorldDataset

        dataset = RealWorldDataset(
            cfgs.env.tasks,
            cfgs.env.cameras,
            cfgs.env.num_episodes,
            data_raw_path=cfgs.env.data_path,
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            mamba_var=cfgs.mamba_var,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            camera_extrinsics=cfgs.env.camera_extrinsics,
            load_processed_data=cfgs.env.load_processed_data,
            save_processed_data=cfgs.env.save_processed_data,
        )

        eval_dataset = RealWorldDataset(
            cfgs.env.tasks,
            cfgs.env.cameras,
            cfgs.env.num_episodes_eval,
            data_raw_path=cfgs.env.data_path,
            traj_len=cfgs.method.horizon,
            frame_skips=cfgs.frame_skips,
            observation_dim=cfgs.method.observation_dim,
            rank_bins=cfgs.method.rank_bins,
            robot=robot,
            mamba_var=cfgs.mamba_var,
            demo_aug_ratio=cfgs.env.demo_aug_ratio,
            demo_aug_min_len=cfgs.env.demo_aug_min_len,
            camera_extrinsics=cfgs.env.camera_extrinsics,
            training=False,
            load_processed_data=cfgs.env.load_processed_data,
            save_processed_data=cfgs.env.save_processed_data,
        )

    trainer = Trainer(
        cfgs=cfgs,
        mamba_model=mamba_model,
        dataset=dataset,
        eval_dataset=eval_dataset,
        train_batch_size=cfgs.batch_size,
        log=cfgs.log,
        log_freq=cfgs.log_freq,
        save_freq=cfgs.save_freq,
        scene_bounds=cfgs.env.scene_bounds,
        project_name=cfgs.project_name,
        online_eval=cfgs.online_eval,
        headless=cfgs.headless,
        rank_bins=cfgs.method.rank_bins,
        robot=robot,
        diffusion_var=cfgs.mamba_var,
        online_eval_start=cfgs.online_eval_start,
        action_mode=cfgs.action_mode,
    )

    trainer.train(cfgs.n_epochs, not cfgs.eval_only)


if __name__ == "__main__":
    main()
