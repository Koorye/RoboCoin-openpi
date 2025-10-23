"""RoboCOIN policy configs."""

import dataclasses
import pathlib
import tyro
from collections.abc import Sequence
from typing import TypeAlias, List, Optional
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.transforms as _transforms


ModelType: TypeAlias = _model.ModelType


def get_robocoin_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import TrainConfig
    import openpi.training.weight_loaders as weight_loaders
    from ..config import DataConfigFactory, DataConfig, ModelTransformFactory
    from ...policies import robocoin_policy

    @dataclasses.dataclass(frozen=True)
    class LeRobotRoboCOINDataConfig(DataConfigFactory):

        delta_action_mask: Optional[List[int]] = None

        repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
            default=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation.images.cam_high": "observation.images.cam_high",
                            "observation.images.cam_left_wrist": "observation.images.cam_left_wrist",
                            "observation.images.cam_right_wrist": "observation.images.cam_right_wrist",
                            "observation.state": "observation.state",
                            "action": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        )

        action_sequence_keys: Sequence[str] = ("action",)

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            data_transforms = _transforms.Group(
                inputs=[robocoin_policy.RoboCOINInputs()],
                outputs=[robocoin_policy.RoboCOINOutputs()],
            )
            if self.delta_action_mask is not None:
                delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
            )

    return [
        # Pi0 w/ RoboCOIN
        TrainConfig(
            name="pi0_robocoin",
            model=pi0_config.Pi0Config(),
            data=LeRobotRoboCOINDataConfig(
                repo_id="robocoin/repo",
                delta_action_mask=None,
            ),
            batch_size=32,
            num_workers=16,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),
        # Pi0 LoRA w/ RoboCOIN
        TrainConfig(
            name="pi0_robocoin_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ),
            data=LeRobotRoboCOINDataConfig(
                repo_id="robocoin/repo",
                delta_action_mask=None,
            ),
            batch_size=32,
            num_workers=16,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Pi0 w/ RoboCOIN (delta actions)
        TrainConfig(
            name="pi0_robocoin_delta",
            model=pi0_config.Pi0Config(),
            data=LeRobotRoboCOINDataConfig(
                repo_id="robocoin/repo",
                delta_action_mask=[7, -1, 7, -1], # 7 left arm joints, skip gripper, 7 right arm joints, skip gripper
            ),
            batch_size=32,
            num_workers=16,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),
        # Pi0 LoRA w/ RoboCOIN (delta actions)
        TrainConfig(
            name="pi0_robocoin_lora_delta",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ),
            data=LeRobotRoboCOINDataConfig(
                repo_id="robocoin/repo",
                delta_action_mask=[7, -1, 7, -1], # 7 left arm joints, skip gripper, 7 right arm joints, skip gripper
            ),
            batch_size=32,
            num_workers=16,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Pi0 LoRA w/ RoboCOIN (debug)
        TrainConfig(
            name="pi0_robocoin_lora_debug",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ),
            data=LeRobotRoboCOINDataConfig(
                repo_id="robocoin/repo",
                delta_action_mask=None,
            ),
            batch_size=1,
            num_workers=1,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=1_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
        ),
    ]

