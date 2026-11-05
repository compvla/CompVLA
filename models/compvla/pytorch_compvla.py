# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The CompVLA Authors. All rights reserved.
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
#
# Modified from LeRobot v0.4.3 (src/lerobot/policies/pi0/modeling_pi0.py) for CompVLA.

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OPENPI_ATTENTION_MASK_VALUE,
)

from models.compvla.configuration_compvla import CompVLAConfig
from models.compvla.paligemma_with_expert_compvla import PaliGemmaWith2ExpertsModel
from models.compvla.utils_compvla import (get_gemma_config, sample_beta, create_sinusoidal_pos_embedding,
                                        make_att_2d_masks, ActionSelectKwargs)
from utils.Lie import Expso3_torch

class CompVLAPytorch(nn.Module):
    """Core CompVLA PyTorch model."""

    def __init__(self, config: CompVLAConfig, rtc_processor: RTCProcessor | None = None, dataset_stats: dict | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWith2ExpertsModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_model_list=config.train_model_list,
            use_compliance_expert=self.config.use_compliance_expert,
        )

        self.action_vel_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_vel_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.action_vel_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_vel_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.vel_dim = self.config.vel_dim
        self.cp_dim = self.config.cp_dim
        self.traj_dim = self.config.traj_dim
        self.joint_dim = self.config.joint_dim
        self.use_joint_in_training = self.config.use_joint_in_training
        # Action stats for unnormalizing velocity before trajectory integration (physical mode only)
        if self.config.traj_condition_space == "physical":
            if dataset_stats is None:
                raise ValueError("traj_condition_space='physical' requires dataset_stats")
            self.register_buffer("vel_mean", torch.as_tensor(dataset_stats[ACTION]["mean"][:self.vel_dim], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.as_tensor(dataset_stats[ACTION]["std"][:self.vel_dim], dtype=torch.float32), persistent=False)
        # Compliance expert part
        self.use_compliance_expert = self.config.use_compliance_expert
        if self.use_compliance_expert:
            self.action_cp_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
            self.action_cp_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
            self.action_cp_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_cp_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)


        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.vel_expert.model.gradient_checkpointing = True
        if self.use_compliance_expert:
            self.paligemma_with_expert.cp_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for CompVLAPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.vel_expert.model.gradient_checkpointing = False
        if self.use_compliance_expert:
            self.paligemma_with_expert.cp_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for CompVLAPytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, expert_type: str):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        if expert_type == "vel_expert":
            action_in_proj = self.action_vel_in_proj
            action_time_mlp_in = self.action_vel_time_mlp_in
            action_time_mlp_out = self.action_vel_time_mlp_out
        elif self.use_compliance_expert and expert_type == "cp_expert":
            action_in_proj = self.action_cp_in_proj
            action_time_mlp_in = self.action_cp_time_mlp_in
            action_time_mlp_out = self.action_cp_time_mlp_out
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")

        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions_vel, actions_cp, noise_vel=None, noise_cp=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise_vel is None:
            noise_vel = self.sample_noise(actions_vel.shape, actions_vel.device)

        if time is None:
            time = self.sample_time(actions_vel.shape[0], actions_vel.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise_vel + (1 - time_expanded) * actions_vel
        u_t = noise_vel - actions_vel

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs_vel, suffix_pad_masks_vel, suffix_att_masks_vel, adarms_cond_vel = self.embed_suffix(state, x_t, time, expert_type='vel_expert')

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs_vel = suffix_embs_vel.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks_vel = torch.cat([prefix_pad_masks, suffix_pad_masks_vel], dim=1)
        att_masks_vel = torch.cat([prefix_att_masks, suffix_att_masks_vel], dim=1)

        att_2d_masks_vel = make_att_2d_masks(pad_masks_vel, att_masks_vel)
        position_ids_vel = torch.cumsum(pad_masks_vel, dim=1) - 1

        att_2d_masks_4d_vel = self._prepare_attention_masks_4d(att_2d_masks_vel)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, expert_type):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                expert_type=expert_type,
            )
            return suffix_out

        suffix_out_vel = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs_vel, att_2d_masks_4d_vel, position_ids_vel, adarms_cond_vel, 'vel_expert'
        )

        suffix_out_vel = suffix_out_vel[:, -self.config.chunk_size :]
        suffix_out_vel = suffix_out_vel.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out, expert_type):
            if expert_type == 'vel_expert':
                return self.action_vel_out_proj(suffix_out)
            elif self.use_compliance_expert and expert_type == 'cp_expert':
                return self.action_cp_out_proj(suffix_out)
            else:
                raise ValueError(f"Unknown expert_type: {expert_type}")

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out_vel, 'vel_expert')
        losses_vel = F.mse_loss(u_t, v_t, reduction="none")
        if not self.use_compliance_expert:
            return losses_vel, None

        # Use compliance expert
        else:
            # get trajectory condition from actions
            traj = self.action_to_traj(actions_vel) # (bsize, chunk_size, traj_dim)

            if actions_cp is None:
                raise ValueError("actions_cp should be provided when training compliance expert")
            if noise_cp is None:
                noise_cp = self.sample_noise(actions_cp.shape, actions_cp.device)
            else:
                noise_cp = noise_cp.clone()

            # trajectory condition
            actions_cp[:, :, self.cp_dim:self.cp_dim + self.traj_dim] = traj
            noise_cp[:, :, self.cp_dim:self.cp_dim + self.traj_dim] = traj

            y_t = time_expanded * noise_cp + (1 - time_expanded) * actions_cp
            p_t = noise_cp - actions_cp

            suffix_embs_cp, suffix_pad_masks_cp, suffix_att_masks_cp, adarms_cond_cp = self.embed_suffix(state, y_t, time, expert_type='cp_expert')

            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs_cp = suffix_embs_cp.to(dtype=torch.bfloat16)

            pad_masks_cp = torch.cat([prefix_pad_masks, suffix_pad_masks_cp], dim=1)
            att_masks_cp = torch.cat([prefix_att_masks, suffix_att_masks_cp], dim=1)

            att_2d_masks_cp = make_att_2d_masks(pad_masks_cp, att_masks_cp)
            position_ids_cp = torch.cumsum(pad_masks_cp, dim=1) - 1

            att_2d_masks_4d_cp = self._prepare_attention_masks_4d(att_2d_masks_cp)

            suffix_out_cp = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs_cp, att_2d_masks_4d_cp, position_ids_cp, adarms_cond_cp, 'cp_expert'
            )

            suffix_out_cp = suffix_out_cp[:, -self.config.chunk_size :]
            suffix_out_cp = suffix_out_cp.to(dtype=torch.float32)

            q_t = self._apply_checkpoint(action_out_proj_func, suffix_out_cp, 'cp_expert')
            losses_cp = F.mse_loss(p_t, q_t, reduction="none")
            return losses_vel, losses_cp

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise_vel=None,
        noise_cp=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device

        if noise_vel is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise_vel = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            expert_type='vel_expert'
        )

        dt = -1.0 / num_steps

        x_t = noise_vel
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call_vel(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    state=state,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    expert_type='vel_expert'
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call_vel,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call_vel(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        if not self.use_compliance_expert:
            return x_t, None
        else:
            # Compliance expert
            # (1) build ee trajectory condition
            traj = self.action_to_traj(x_t) # (bsize, chunk_size, traj_dim)

            # (2) denoise with compliance expert
            if noise_cp is None:
                # Sample noise with padded dimension as expected by action_in_proj
                actions_shape = (
                    bsize,
                    self.config.chunk_size,
                    self.config.max_action_dim,
                )  # Use config max_action_dim for internal processing
                noise_cp = self.sample_noise(actions_shape, device)
            else:
                noise_cp = noise_cp.clone()
            noise_cp[:, :, self.cp_dim:self.cp_dim + self.traj_dim] = traj

            y_t = noise_cp
            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

                # RTC is not supported with the compliance expert (see CompVLAConfig)
                v_t = self.denoise_step(
                    state=state,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=y_t,
                    timestep=time_tensor,
                    expert_type='cp_expert'
                )

                y_t = y_t + dt * v_t

                # Fix the trajectory part
                y_t[:, :, self.cp_dim:self.cp_dim + self.traj_dim] = traj

            return x_t, y_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        expert_type: str,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if expert_type == "vel_expert":
            current_expert = self.paligemma_with_expert.vel_expert
            action_out_proj = self.action_vel_out_proj
        elif self.use_compliance_expert and expert_type == "cp_expert":
            current_expert = self.paligemma_with_expert.cp_expert
            action_out_proj = self.action_cp_out_proj
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep, expert_type=expert_type)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        current_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            expert_type=expert_type
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return action_out_proj(suffix_out)


    def action_to_traj(self, x_t: torch.Tensor):
        """Integrate the velocity chunk into a trajectory relative to the current pose, plus gripper."""
        bsize = x_t.shape[0]
        vel = x_t[:, :, :self.vel_dim].detach().clone() # (bsize, chunk_size, 7)
        if self.config.traj_condition_space == "physical":
            vel = vel * self.vel_std + self.vel_mean
        traj = x_t.new_zeros((bsize, self.config.chunk_size, 4, 4), requires_grad=False)
        dt = 1.0 / self.config.fps
        traj[:, :, 0, 0] = traj[:, :, 1, 1] = traj[:, :, 2, 2] = traj[:, :, 3, 3] = 1.0
        for idx in range(vel.shape[1]-1):
            traj[:, idx+1, :3, :3] = torch.einsum('nij, njk -> nik', traj[:, idx, :3, :3], Expso3_torch(vel[:, idx, :3] * dt))
            traj[:, idx+1, :3, 3] = traj[:, idx, :3, 3] + torch.einsum('nij, nj -> ni', traj[:, idx, :3, :3], vel[:, idx, 3:6] * dt)

        traj = traj[:, :, :3, :4].reshape(bsize, self.config.chunk_size, 12)  # (bsize, chunk_size, 12)
        traj = torch.cat([traj, vel[:, :, 6:7]], dim=-1)  # (bsize, chunk_size, 13)
        assert traj.shape[-1] == self.traj_dim, "traj shape should be (bsize, chunk_size, traj_dim)"

        return traj
