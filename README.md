# RoboCOIN OpenPI

[English](README.md) | [中文](README_zh-CN.md) | [官方OpenPI文档](READMD_openpi.md)

> ⚠️ ​This repository is still under development and has not yet integrated the latest CoRobot format.​​ ⚠️


## Overview
RoboCOIN OpenPI is a VLA (Vision-Language-Action) training codebase based on the OpenPI model training framework. It has been extensively optimized for the RoboCOIN dataset, with enhancements in data loading and training strategies.

​Core Features:​​
- ​Dataset Support: Integrated RoboCOIN dataset with support for fine-grained text descriptions.
- ​Training Strategies: Implemented training methods and optimization strategies tailored for RoboCOIN tasks.

## Installation

1. Prerequisites

    Official Requirements:

   | Mode | Memory Requirement | GPU Examples |
   |-------|------------------|----------------|
   | Inference | > 8 GB | RTX 4090 |
   | Fine-tuning (LoRA) | > 22.5 GB | RTX 4090 |
   | Fine-tuning (Full) | > 70 GB | A100 (80GB) / H100 |

   ⚠️Actual testing shows LoRA fine-tuning requires over 32GB of VRAM!​​ 

2. Clone the Repository
   ```bash
   git clone --recurse-submodules https://github.com/Koorye/RoboCoin-openpi.git
   ```

3. Install Dependencies

   Refer to the [OpenPI Official Repository](https://github.com/Physical-Intelligence/openpi) for detailed instructions.
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   ```

4. Download Pre-trained Model Weights

   RoboCOIN OpenPI uses OpenPI's pre-trained models as a foundation. Refer to the [OpenPI Official Repository](https://github.com/Physical-Intelligence/openpi) for more information.

   | Model | Description | Download Link |
   |-------|-------------|----------------|
   | $\pi_0$ | Base model for fine-tuning | gs://openpi-assets/checkpoints/pi0_base |
   | $\pi_0$-FAST | Autoregressive version for fine-tuning | gs://openpi-assets/checkpoints/pi0_fast_base |
   | $\pi_{0.5}$ | Enhanced version with hierarchical reasoning | gs://openpi-assets/checkpoints/pi05_base |

## Model Configuration

RoboCOIN OpenPI offers various preset configurations to suit different training needs:

| Configuration Name | Batch Size | Learning Rate | Training Steps | Lora | Relative Action |
|----------|------------|--------|----------|------|----------|
| pi0_robocoin | 32        | 2.5e-5   | 30k     | ❌   | ❌       |
| pi0_robocoin_lora | 32        | 2.5e-5   | 30k     | ✅   | ❌       |
| pi_robocoin_delta | 32        | 2.5e-5   | 30k     | ❌   | ✅       |
| pi0_robocoin_delta_lora | 32        | 2.5e-5   | 30k     | ✅   | ✅       |
| pi0_robocoin_lora_debug | 1      | 2.5e-5   | 1000     | ✅   | ❌       |

## Model Usage

1. Data Statistics

   It is recommended to compute normalization statistics for the dataset before starting training.

   OpenPI Official Script:
   ```bash
   uv run scripts/compute_norm_stats.py \
       --config-name <config_name>
   ```

   Fast Statistics Script (Recommended):
   ```bash
   uv run scripts/compute_norm_stats_jax.py \
       --config-name <config_name> \
       --use-fast-stats [--no-use-fast-stats] \
       --per-device-batch <batch_size> \
       --num-workers <num_workers>
   ```
   *💡Tip: It is recommended to use batch_size <= 128and num_workers <= 16. Setting these values too high may cause performance issues.*

   Parameter Explanation:​
   | Parameter Name       | Type    | Description                       | Default Value |
   |----------------|-------|--------------------------------|-----------|
   | --config-name  | str   | Configuration Name                   | None      |
   | --use-fast-stats | bool  | Whether to use fast statistics method | False     |
   | --per-device-batch | int   | Per-device batch size                | 128       |
   | --num-workers  | int   | Number of worker threads for data loading | 16        |

2. Model Training:
    ```bash
    uv run train.py \
        <config_name> \
        --exp-name=<experiment_name> \
        --wandb-enabled [--no-wandb-enabled] \
        [--overwrite] \
        [--check-only] \
        [--use-annotation] \
        --use-indices=<indices>
    ```

    Parameter Explanation：
    | Parameter Name | Type | Description | Default Value |
    |-------|------|------------------------------------------|--------------|
    | --wandb-enabled | bool | Whether to enable Weights & Biases for experiment tracking | True          |
    | --overwrite    | bool | Whether to overwrite existing weights                      | False       |
    | --check-only | bool | Whether to only check configuration without training               | False        |
    | --use-annotation | bool | Whether to use fine-grained text descriptions for training              | False          |
    | --use-indices  | List[Tuple(int, int)] | Specify the data index range to use, formatted as "[(start1,end1),(start2,end2)]" or None, where None means to use all data | None |

3. Model Inference
    ```bash
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=<config_name> \
        --policy.dir=/path/to/checkpoint \
        --port=<port>
    ```

    Parameter Explanation:
    | Parameter Name | Type | Description | Default Value |
    |-------|------|-------------------------------|--------------|
    | --policy.config | str  | Model Configuration Name                     | None           |
    | --policy.dir    | str  | Model Checkpoint Path                   | None           |
    | --port          | int  | Server Port Number                       | 8080         |

    Client Pseudocode Example:
    ```python
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    robot = Robot() # 假设有一个Robot类用于机器人控制
    policy = WebsocketClientPolicy(
        host=host,
        port=port,
    )

    # 假设获取到图像和状态向量
    observation = {
        'observation.images.cam_high': image_high, # (H, W, 3) np.ndarray
        'observation.images.cam_left_wrist': image_left_wrist, # (H, W, 3) np.ndarray
        'observation.images.cam_right_wrist': image_right_wrist, # (H, W, 3) np.ndarray
        'observation.state': state_vector, # (N,) np.ndarray
        'prompt': "your text prompt here",
    }
    
    # 获取动作序列并执行
    actions = policy.infer(observation)['action'] # (chunk_size, action_dim) np.ndarray
    for action in actions:
        # 执行动作
        robot.execute_action(action) 
    ```

## Frequently Asked Questions (FAQ)

| Question | Answer |
|------|------|
| CUDA out of memory error despite meeting VRAM requirements | It is recommended to configure [JAX memory allocation](https://jax.net.cn/en/latest/gpu_memory_allocation.html). Use XLA_PYTHON_CLIENT_PREALLOCATE=false to disable preallocation behavior. | 
| Program exits without any error messages | This is often due to internal JAX errors, which may be caused by insufficient disk, memory, or VRAM. You can check the system logs at /var/log/syslog for more information. |

## Acknowledgements

Thanks to the [OpenPI](https://github.com/Physical-Intelligence/openpi) team for providing the powerful model training framework!