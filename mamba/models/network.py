import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from mamba_ssm import Mamba

from mamba.models.helpers import (
    Losses,
    apply_conditioning,
    cosine_beta_schedule,
    extract,
)
import mamba.utils as utils
from mamba.robot import MambaRobot

from mamba.models.pointnet import PointNetfeat
from mamba.models.resnet import ResnetEncoder

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(
        self,
        horizon: int,
        observation_dim: int,
        dim_mults: list,
        action_dim: int,
        scene_bounds: list,
        joint_limits: list,
        n_timesteps: int,
        loss_type: str,
        clip_denoised: bool,
        predict_epsilon: bool,
        hidden_dim: int,
        loss_discount: float,
        condition_guidance_w: float,
        reverse_train: bool,
        conditions: list,
        hard_conditions: list,
        noise_init_method: str,
        loss_fn: str,
        coverage_weight: float,
        detach_reverse: bool,
        joint_weight: float,
        robot_offset: list,
        trans_loss_scale: float,
        rot_loss_scale: float,
        mamba_var: str,
        joint_pred_pose_loss: bool,
        joint_loss_scale: float,
        rank_bins: int,
        backbone: str,
        num_decoder_layers: int,
        num_encoder_layers: int,
        n_head: int,
        causal_attn: bool,
        depth_proc: str,
        rgb_encoder: str,
        **kwargs,
    ):
        super(SimpleModel, self).__init__()

        self._horizon = horizon
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._transition_dim = observation_dim
        self._condition_guidance_w = condition_guidance_w
        self._diffusion_var = mamba_var
        self._noise_init_method = noise_init_method
        self._detach_reverse = detach_reverse
        self._hidden_dim = hidden_dim
        self._joint_weight = joint_weight
        self._joint_pred_pose_loss = joint_pred_pose_loss

        self._trans_loss_scale = trans_loss_scale
        self._rot_loss_scale = rot_loss_scale
        self._joint_loss_scale = joint_loss_scale

        self._proprio_dim = 24 #FIXME
        self._rank_bins = rank_bins

        self._conditions = conditions
        self._reverse_train = (
            reverse_train
            and mamba_var == "gripper_poses"
            and "end" in self._conditions
        )
        self._hard_conditions = hard_conditions
        self._condition_dropout = 0.15 #FIXME
        self._mask_dist = Bernoulli(probs=1 - self._condition_dropout)

        robot_offset = torch.FloatTensor(robot_offset)
        scene_bounds = torch.FloatTensor(scene_bounds)
        joint_limits = torch.FloatTensor(joint_limits)

        # Setup for condition embedding
        condition_fns = {}
        mlp_shape_mapping = {
            "return": 1,
            "start": 7,
            "end": 7,
            "proprios": self._proprio_dim,
            "rank": self._rank_bins,
        }

        dim = hidden_dim
        embed_dim = 0
        act_fn = nn.Mish()

        for cond_name in self._conditions:
            if cond_name == "pcds":
                if depth_proc == "pointnet":
                    condition_fns[cond_name] = nn.Sequential(
                        PointNetfeat(),
                        nn.Mish(),
                        nn.Linear(1024, 256),
                        nn.Mish(),
                        nn.Linear(256, dim),
                    )
                else:
                    raise NotImplementedError
            elif cond_name == "rgbs":
                resnet = ResnetEncoder(
                    rgb=True, freeze=False, pretrained=True, model=rgb_encoder
                )
                condition_fns[cond_name] = nn.Sequential(
                    resnet,
                    nn.Mish(),
                    nn.Linear(resnet.n_channel, dim),
                )
            else:
                condition_fns[cond_name] = nn.Sequential(
                    nn.Linear(mlp_shape_mapping[cond_name], dim),
                    act_fn,
                    nn.Linear(dim, dim * 4),
                    act_fn,
                    nn.Linear(dim * 4, dim),
                )
        self._cond_fns = nn.ModuleDict(condition_fns)
        embed_dim += (len(self._conditions) - 1) * dim

        # Setup mamba layers for predict joint and angle
        self.joint_embedding_layer = nn.Sequential(
            nn.Linear(embed_dim, 7),
        )

        self.joint_mamba_layer = nn.Sequential(
            Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=7, # Model dimension d_model
                d_state=128,  # SSM state expansion factor, typically 64 or 128
                d_conv=8,    # Local convolution width
                expand=2,    # Block expansion factor
            ),
        )

        self.pose_embedding_layer = nn.Sequential(
            nn.Linear(7, 7),
        )

        self.pose_mamba_layer = nn.Sequential(
            Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=7, # Model dimension d_model
                d_state=128,  # SSM state expansion factor, typically 64 or 128
                d_conv=8,    # Local convolution width
                expand=2,    # Block expansion factor
            ),
        )

        # Initialize for loss function
        loss_weights = self.get_loss_weights(loss_discount)

        self._trans_loss_scale = trans_loss_scale
        if loss_fn == "state_chamfer":
            self._loss_fn = Losses[loss_fn](loss_weights, coverage_weight, loss_type)
        else:
            self._loss_fn = Losses[loss_fn](loss_weights)

    def proc_cond(self, cond: dict) -> dict:
        """
        Process the given condition dictionary.

        Parameters:
            cond (dict): The condition dictionary to be processed.

        Returns:
            dict: The processed condition dictionary.
        """
        new_cond = {k: v for k, v in cond.items()}

        if -1 in new_cond and "end" not in self._conditions:
            del new_cond[-1]

        return new_cond

    def get_loss_weights(self, discount: float) -> torch.Tensor:
        """
        Sets loss coefficients for trajectory.

        Args:
            discount (float): The discount factor.

        Returns:
            torch.Tensor: The loss weights for each timestep and dimension.
        """
        dim_weights = torch.ones(self._observation_dim, dtype=torch.float32)

        # Decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self._horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)

        return loss_weights

    def gripper_pose_loss(self, p1, p2):
        trans, quat = p1[..., :3], p1[..., 3:]
        trans_recon, quat_recon = p2[..., :3], p2[..., 3:]

        trans_loss = F.mse_loss(trans, trans_recon) * self._trans_loss_scale
        rot_loss = (
            utils.geodesic_distance_between_quaternions(quat, quat_recon).mean()
            * self._rot_loss_scale
        )

        info = {
            "trans_loss": trans_loss,
            "rot_loss": rot_loss,
        }

        loss = trans_loss + rot_loss

        return loss, info

    def forward(
        self,
        cond: dict,
        horizon: int = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        *args,
        **kwargs
    ):
        cond = self.proc_cond(cond)

        start_emb = []
        end_emb = []

        for cond_name in sorted(self._conditions):
            cond_var = kwargs[cond_name]
            if cond_name == "rgbs":
                cond_var = cond_var.permute(0, 3, 1, 2)

            cond_emb = self._cond_fns[cond_name](cond_var)
            allow_dropout = cond_name not in self._hard_conditions
            if use_dropout and allow_dropout:
                mask = self._mask_dist.sample(sample_shape=(cond_emb.size(0), 1)).to(
                    cond_emb.device
                )
                cond_emb = mask * cond_emb
            if force_dropout and allow_dropout:
                cond_emb = 0 * cond_emb

            if cond_name == "start":
                start_emb.append(cond_emb)
            
            elif cond_name == "end":
                end_emb.append(cond_emb)

            if cond_name != "start" and cond_name != "end":
                start_emb.append(cond_emb)
                end_emb.append(cond_emb)
        
        start_joint = torch.cat(start_emb, dim=-1)
        end_joint = torch.cat(end_emb, dim=-1)

        start_joint = self.joint_embedding_layer(start_joint)
        end_joint = self.joint_embedding_layer(end_joint)

        batch_size = start_joint.size(0)
        t = torch.linspace(0, 1, steps=self._horizon, device=start_joint.device).unsqueeze(0).unsqueeze(-1)  # Shape (1, self._horizon, 1)
        interpolated_joint = start_joint.unsqueeze(1) * (1 - t) + end_joint.unsqueeze(1) * t  # Shape (batch_size, self._horizon, 7)

        joint_feats = self.joint_mamba_layer(interpolated_joint)

        pose_emb = self.pose_embedding_layer(interpolated_joint)
        pose_emb = self.pose_mamba_layer(pose_emb)
        
        return joint_feats, pose_emb

    def conditional_sample(
        self, cond: dict, horizon: int = None, *args, **kwargs
    ) -> dict:
        """
        A function that performs a conditional sample of the diffusion model.

        Args:
            cond (dict): The conditions for the sample.
            horizon (int, optional): The horizon for the sample. Defaults to None.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the sampled values.
        """
        cond = self.proc_cond(cond)
        batch_size = len(cond[0])
        horizon = horizon or self._horizon
        shape = (batch_size, horizon, self._observation_dim)

        joint_emb, gripper_emb = self(cond, horizon, use_dropout=False, force_dropout=False, *args, **kwargs)
        return_dict = {"traj": joint_emb}

        if self._diffusion_var == "joint_positions":
            seq_len = shape[1]
            robot = kwargs["robot"]
            joints = joint_emb.view(-1, 7)
            predicted_traj = robot.forward_kinematics_batch(joints).view(
                batch_size, seq_len, 7
            )

            return_dict["traj"] = predicted_traj
            return_dict["joint_positions"] = joint_emb
            return_dict["gripper_poses"] = gripper_emb
        return {self._diffusion_var: return_dict}

    def loss(self, x: torch.tensor, cond: dict, robot: MambaRobot = None, **kwargs) -> tuple:
        x_start = x
        gt_poses = kwargs["gripper_poses"]
        conditions = kwargs
        
        if self._reverse_train:
            reversed_cond = {-1: cond[0], 0: cond[-1]}
            cond = {k: torch.cat([v, reversed_cond[k]], dim=0) for k, v in cond.items()}

            reversed_start = torch.flip(x_start, dims=(1,))
            x_start = torch.cat([x_start, reversed_start], dim=0)

            reversed_gt_poses = torch.flip(gt_poses, dims=(1,))
            gt_poses = torch.cat([gt_poses, reversed_gt_poses], dim=0)

            new_conditions = {}

            for k, v in conditions.items():
                if k == "start":
                    new_conditions[k] = torch.cat(
                        [conditions["start"], conditions["end"]], dim=0
                    )
                elif k == "end":
                    new_conditions[k] = torch.cat(
                        [conditions["end"], conditions["start"]], dim=0
                    )
                else:
                    new_conditions[k] = v.repeat(
                        2, *[1 for _ in range(len(v.shape[1:]))]
                    )

            conditions = new_conditions

        x_recon, predicted_poses = self(cond, horizon=self._horizon, use_dropout=True, force_dropout=False, **conditions)

        batch_size = x.size(0)

        loss, _ = self._loss_fn(x_start, x_recon)
        loss = loss * self._joint_loss_scale
        info = {"joint_loss": loss}

        if self._joint_pred_pose_loss:
            assert robot is not None
            # f_poses = robot.forward_kinematics_batch(
            #     x_recon.contiguous().view(-1, 7)
            # ).view(batch_size, -1, 7)

            # pose_loss = F.mse_loss(predicted_poses[..., :3], gt_poses[..., :3])
            pose_loss, _ = self._loss_fn(predicted_poses, gt_poses)
            pose_loss = pose_loss * self._trans_loss_scale

            loss += pose_loss
            info["pose_loss"] = pose_loss

        return loss, info