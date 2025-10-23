# RoboCOIN OpenPI

[English](README.md) | [中文](README_zh-CN.md)

> ⚠️⚠️⚠️本仓库仍在开发中，尚未接入最新的CoRobot格式⚠️⚠️⚠️

## 概述

RoboCOIN OpenPI​ 是一个基于 OpenPI模型训练框架扩展的 VLA（Vision-Language-Action）训练代码库，针对 RoboCOIN 数据集的需求进行了深度优化，扩展了数据读取与训练策略

核心功能：
- 数据集支持：集成RoboCOIN数据集，支持细粒度文本描述
- 训练策略：实现适用于RoboCOIN任务的训练方法和优化策略

## 安装

1. 前置需求

    官方需求：

   | 模式 | 显存需求 | GPU示例 |
   |-------|------------------|----------------|
   | 推理	| > 8 GB | RTX 4090 |
   | 微调 (LoRA) | > 22.5 GB | RTX 4090 |
   | 微调 (全量) | > 70 GB | A100 (80GB) / H100 |

   ⚠️实际测试LoRA微调需要32GB以上显存！

2. 克隆仓库
   ```bash
   git clone --recurse-submodules https://github.com/Koorye/RoboCoin-openpi.git
   ```

3. 安装依赖

   参照[OpenPI官方仓库](https://github.com/Physical-Intelligence/openpi)
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   ```

4. 预训练模型权重下载

   RoboCOIN OpenPI使用OpenPI的预训练模型作为基础，参考[OpenPI官方仓库](https://github.com/Physical-Intelligence/openpi)

   | 模型 | 描述 | 下载链接 |
   |-------|------------------|----------------|
   | $\pi_0$ | 用于微调的基础模型 | gs://openpi-assets/checkpoints/pi0_base |
   | $\pi_0$-FAST | 自回归版本的微调基础模型 | gs://openpi-assets/checkpoints/pi0_fast_base |
   | $\pi_{0.5}$ | 加入分层推理的微调基础模型 | gs://openpi-assets/checkpoints/pi05_base |

## 模型配置

RoboCOIN OpenPI 提供多种预设配置，适应不同训练需求：

| 配置名称 | Batch Size | 学习率 | 训练步数 | Lora | 相对动作 | 
|----------|------------|--------|----------|------|----------|
| pi0_robocoin | 32        | 2.5e-5   | 30k     | ❌   | ❌       |
| pi0_robocoin_lora | 32        | 2.5e-5   | 30k     | ✅   | ❌       |
| pi_robocoin_delta | 32        | 2.5e-5   | 30k     | ❌   | ✅       |
| pi0_robocoin_delta_lora | 32        | 2.5e-5   | 30k     | ✅   | ✅       |
| pi0_robocoin_lora_debug | 1      | 2.5e-5   | 1000     | ✅   | ❌       |

## 模型使用

1. 数据统计：在开始训练前，建议先计算数据集的归一化统计信息

   OpenPI官方脚本：
   ```bash
   uv run scripts/compute_norm_stats.py \
       --config-name <config_name>
   ```

   快速统计脚本（推荐）：
   ```bash
   uv run scripts/compute_norm_stats_jax.py \
       --config-name <config_name> \
       --use-fast-stats [--no-use-fast-stats] \
       --per-device-batch <batch_size> \
       --num-workers <num_workers>
   ```
   *💡​提示​：建议使用batch_size<=128, num_workers<=16，设置过大可能导致卡顿*

   参数说明：
    | 参数名           | 类型    | 说明                             | 默认值      |
    |----------------|-------|--------------------------------|-----------|
    | --config-name  | str   | 配置名称                           | 无         |
    | --use-fast-stats | bool  | 是否使用快速统计方法                    | False     |
    | --per-device-batch | int   | 每个设备的批量大小                     | 128        |
    | --num-workers  | int   | 数据加载的工作线程数                   | 16         |

2. 模型训练：
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

    参数说明：
    | 参数名 | 类型 | 说明 | 默认值 |
    |-------|------|------------------------------------------|--------------|
    | --wandb-enabled | bool | 是否启用Weights & Biases进行实验跟踪          | True          |
    | --overwrite    | bool | 是否覆盖已有有权重                      | False       |
    | --check-only | bool | 是否仅进行配置检查，不执行训练               | False        |
    | --use-annotation | bool | 是否使用细粒度文本描述进行训练              | False          |
    | --use-indices  | List[Tuple(int, int)] | 指定使用的数据索引范围，格式为"[(start1,end1),(start2,end2)]"或None，None表示使用全部数据 | None |

3. 模型推理
    ```bash
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=<config_name> \
        --policy.dir=/path/to/checkpoint \
        --port=<port>
    ```

    参数解释：
    | 参数名 | 类型 | 说明 | 默认值 |
    |-------|------|-------------------------------|--------------|
    | --policy.config | str  | 模型配置名称                     | 无           |
    | --policy.dir    | str  | 模型检查点路径                   | 无           |
    | --port          | int  | 服务端口号                       | 8080         |

    客户端伪代码示例：
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

## 常见问题

| 问题 | 解答 |
|------|------|
| 显存符合要求但报错CUDA out of memory | 建议配置[JAX内存分配](https://jax.net.cn/en/latest/gpu_memory_allocation.html)，使用`XLA_PYTHON_CLIENT_PREALLOCATE=false`关闭预分配行为 | 
| 程序退出，但无任何报错 | jax内部错误，往往是因为硬盘、内存或显存不足，你可能可以通过系统日志/var/log/syslog查看详细信息 |

## 致谢

感谢[OpenPI](https://github.com/Physical-Intelligence/openpi)团队提供的强大模型训练框架！