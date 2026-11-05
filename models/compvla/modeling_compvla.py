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

import builtins
import logging
from collections import deque
from pathlib import Path

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from models.compvla.configuration_compvla import CompVLAConfig
from models.compvla.pytorch_compvla import CompVLAPytorch
from models.compvla.utils_compvla import resize_with_pad_torch, pad_vector, pad_vector_cp, ActionSelectKwargs

class CompVLAPolicy(PreTrainedPolicy):
    """CompVLA Policy for LeRobot."""

    config_class = CompVLAConfig
    name = "compvla"

    def __init__(
        self,
        config: CompVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.use_compliance_expert = self.config.use_compliance_expert
        self.vel_dim = self.config.vel_dim
        self.cp_dim = self.config.cp_dim
        self.traj_dim = self.config.traj_dim
        self.joint_dim = self.config.joint_dim
        self.use_joint_in_training = self.config.use_joint_in_training

        # Initialize the core CompVLA model
        self.init_rtc_processor()
        self.model = CompVLAPytorch(config, rtc_processor=self.rtc_processor, dataset_stats=kwargs.get("dataset_stats"))

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,  # upstream PI0 uses True; False lets a PI0 checkpoint load before the expert remapping below
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "CompVLA is built on PI0, a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0
            not_remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value
                    not_remap_count += 1

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys (unchanged : {not_remap_count})")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            # PaliGemma ties embed_tokens to lm_head (one shared tensor), so loading lm_head also fills embed_tokens
            embed_tokens_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
            if embed_tokens_key in missing_keys and lm_head_key in remapped_state_dict:
                missing_keys.remove(embed_tokens_key)

            # Initialize the CompVLA experts from a PI0 checkpoint: keys missing for vel_expert / cp_expert
            # (and their action_vel_* / action_cp_* projections) are copied from PI0's gemma_expert
            # (and action_* projections). Keys that already exist in the checkpoint are loaded as-is.
            model_state_dict = model.state_dict()
            if config.use_compliance_expert:
                for key in list(missing_keys):
                    if key.split('.')[2] == 'cp_expert':
                        corresponding_key = key.replace('cp_expert', 'gemma_expert')
                        if corresponding_key in remapped_state_dict:
                            model_state_dict[key].data.copy_(remapped_state_dict[corresponding_key].data)
                            missing_keys.remove(key)
                        else:
                            print(f"No corresponding key for {key}")
                    elif key.split('.')[1].startswith('action_cp'):
                        corresponding_key = key.replace('action_cp', 'action')
                        if corresponding_key in remapped_state_dict:
                            model_state_dict[key].data.copy_(remapped_state_dict[corresponding_key].data)
                            missing_keys.remove(key)
                        else:
                            print(f"No corresponding key for {key}")
            for key in list(missing_keys):
                if key.split('.')[2] == 'vel_expert':
                    corresponding_key = key.replace('vel_expert', 'gemma_expert')
                    if corresponding_key in remapped_state_dict:
                        model_state_dict[key].data.copy_(remapped_state_dict[corresponding_key].data)
                        missing_keys.remove(key)
                        if corresponding_key in unexpected_keys:
                            unexpected_keys.remove(corresponding_key)
                        else:
                            print(f"Failed to remove {corresponding_key}")
                    else:
                        print(f"No corresponding key for {key}")
                elif key.split('.')[1].startswith('action_vel'):
                    corresponding_key = key.replace('action_vel', 'action')
                    if corresponding_key in remapped_state_dict:
                        model_state_dict[key].data.copy_(remapped_state_dict[corresponding_key].data)
                        missing_keys.remove(key)
                        if corresponding_key in unexpected_keys:
                            unexpected_keys.remove(corresponding_key)
                        else:
                            print(f"Failed to remove {corresponding_key}")
                    else:
                        print(f"No corresponding key for {key}")

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.vel_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.vel_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi0
            # non-pi05 model expects action_time_mlp_*, but checkpoint might have time_mlp_*
            if key.startswith("time_mlp_in."):
                new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."):
                new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        full_actions = batch[ACTION]
        if self.use_joint_in_training:
            action_vel = torch.cat([full_actions[..., :self.vel_dim],
                                    full_actions[..., self.vel_dim + self.cp_dim:self.vel_dim + self.cp_dim + self.joint_dim]], dim=-1)
        else:
            action_vel = full_actions[..., :self.vel_dim]
        actions_vel = pad_vector(action_vel, self.config.max_action_dim)

        if not self.use_compliance_expert:
            return actions_vel, None
        else:
            if self.use_joint_in_training:
                action_cp = full_actions[..., self.vel_dim:self.vel_dim + self.cp_dim + self.joint_dim]
                actions_cp = pad_vector_cp(action_cp, self.config.max_action_dim, self.cp_dim, self.traj_dim, self.joint_dim)
            else:
                action_cp = full_actions[..., self.vel_dim:self.vel_dim + self.cp_dim]
                actions_cp = pad_vector(action_cp, self.config.max_action_dim)
            return actions_vel, actions_cp

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)

        # Sample actions using the model (pass through RTC kwargs)
        actions_vel, actions_cp = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0] # == data_action_dim = 7+12+7 = 26
        actions = actions_vel[:, :, :self.vel_dim]
        if self.use_compliance_expert:
            actions_cp = actions_cp[:, :, :self.cp_dim]
            padding_dim = original_action_dim - self.vel_dim - self.cp_dim
            actions_zero = torch.zeros(actions_vel.shape[:-1] + (padding_dim, )).to(actions)
            actions = torch.cat([actions, actions_cp, actions_zero], dim=-1)
        else:
            padding_dim = original_action_dim - self.vel_dim
            actions_zero = torch.zeros(actions_vel.shape[:-1] + (padding_dim, )).to(actions)
            actions = torch.cat([actions, actions_zero], dim=-1)

        return actions # (batch_size, chunk_size, vel_dim + cp_dim + zero_padding) -> matching dimension with postprocessor(unnormalizer)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        actions_vel, actions_cp = self.prepare_action(batch)

        # Compute loss
        loss_dict = {}
        losses_vel, losses_cp = self.model.forward(images, img_masks, lang_tokens, lang_masks, state,
                                                    actions_vel=actions_vel, actions_cp=actions_cp)

        # Truncate losses to actual action dimensions
        if self.use_joint_in_training:
            losses_vel = losses_vel[:, :, :self.vel_dim + self.joint_dim]
        else:
            losses_vel = losses_vel[:, :, :self.vel_dim]
        loss_dict["loss_vel"] = losses_vel.mean().item()
        if not self.use_compliance_expert:
            losses = losses_vel
        else:
            if self.use_joint_in_training:
                losses_cp_pure = losses_cp[:, :, : self.cp_dim]
                losses_joint = losses_cp[:, :, self.cp_dim + self.traj_dim : self.cp_dim + self.traj_dim + self.joint_dim]
                losses_cp = torch.cat([losses_cp_pure, losses_joint], dim=-1)
            else:
                losses_cp = losses_cp[:, :, : self.cp_dim]
            loss_dict["loss_cp"] = losses_cp.mean().item()
            losses = torch.cat([losses_vel, losses_cp], dim=-1)

        loss_dict["loss_per_dim"] = losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for CompVLA fine-tuning."""
        vel_common_projections = (
            "state_proj|action_vel_in_proj|action_vel_out_proj|action_vel_time_mlp_in|action_vel_time_mlp_out"
        )
        vel_target_modules = rf"(.*\.vel_expert\..*\.self_attn\.(q|v)_proj|model\.({vel_common_projections}))"
        target_modules = vel_target_modules
        if self.use_compliance_expert:
            cp_common_projections = (
                "action_cp_in_proj|action_cp_out_proj|action_cp_time_mlp_in|action_cp_time_mlp_out"
            )
            cp_target_modules = rf"(.*\.cp_expert\..*\.self_attn\.(q|v)_proj|model\.({cp_common_projections}))"
            target_modules = rf"({vel_target_modules}|{cp_target_modules})"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
