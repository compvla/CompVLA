# CompVLA

**CompVLA: A Variable Compliance Vision-Language-Action Model for Contact-rich Manipulation** (CoRL 2026)

[Project page](https://compvla.github.io) | [Paper](https://arxiv.org/abs/2609.23614)

CompVLA extends a VLA (PI0) with a Compliance Expert that predicts time-varying stiffness and virtual displacement alongside motion, executed through geometric impedance control.
This repository contains the CompVLA model, the training code, and the conversion code from our raw robot data (Franka FR3) to LeRobot datasets.

## Installation

The code is built on [LeRobot](https://github.com/huggingface/lerobot) v0.4.3.

### Create environment
```bash
conda create -y -n compvla python=3.11
conda activate compvla
conda install -y ffmpeg=7.1.1 -c conda-forge
```

### Install packages
- RTX 5090 (sm_120), which needs the CUDA 12.8 wheels:
```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128
pip install -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 "lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.4.3"
pip install -r requirements_extra.txt
```

- Other GPUs (RTX 4090, ...):
```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.4.3"
pip install -r requirements_extra.txt
```

- (Optional) PEFT, only needed for [training with PEFT](#training-with-peft-experimental):
```bash
python -m pip install "peft>=0.18.0,<1.0.0" --no-deps
```

### Hugging Face login
The PaliGemma tokenizer (`google/paligemma-3b-pt-224`) is gated. Accept its license on the model page, then log in:
```bash
huggingface-cli login
```

### PI0 checkpoint
CompVLA is initialized from PI0. Download `lerobot/pi0_base` at the revision below; later revisions add processor steps that LeRobot v0.4.3 cannot load.
```bash
huggingface-cli download lerobot/pi0_base --revision 26b99b9439acb1e352439e34ee9c67af0d76efa3 --local-dir checkpoints/pi0
```

## Dataset

### Download
Download the raw data (one zip file per task, 5.8 GB in total and 7.1 GB unzipped) from [Google Drive](https://drive.google.com/drive/folders/15SLhbUN81K4f6syi3K83eudc17dwK3hx) and unzip it into `datasets/raw_data`:
```bash
mkdir -p datasets/raw_data
for TASK in Erase Pivot PnP Dispense; do unzip ${TASK}.zip -d datasets/raw_data; done
```
This gives `datasets/raw_data/<Task>/<episode>/`, where each episode folder (named by its recording time) contains:
- `right_arm_joints.csv`, `left_arm_joints.csv`, `right_arm_gripper.csv`, `ft_sensor_wrench.csv`
- `cam{0,1,2}_rgb.avi` (640x480) and `cam{0,1,2}_rgb_timestamps.csv`

| Task | Episodes | Episode length |
|---|---|---|
| `Erase` | 107 | 26-56 s |
| `Pivot` | 115 | 18-38 s |
| `PnP` | 109 | 19-32 s |
| `Dispense` | 111 | 18-32 s |

The conversion code uses the first 100 episodes of each task in sorted folder order (`num_episodes` in `datasets_convert/raw_to_lerobot.yaml`).

### Raw data notes
- `left_arm_joints.csv` is the leader arm (teleoperation) and `right_arm_joints.csv` is the follower arm (the robot). The F/T sensor (`ft_sensor_wrench.csv`) is mounted on the follower.
- The teleoperation setup differs across tasks, so the leader-follower relation is not consistent between tasks. Use the follower (right arm) for actions; the conversion code uses the right arm only.
- Sampling intervals are irregular and differ between files; align streams by their `time` column (the conversion code resamples everything to a common timeline).
- The value order in `positions` follows the `joint_names` column, which is not always `joint1..7`.

### Convert to LeRobot
Edit `datasets_convert/raw_to_lerobot.yaml` if needed, then run from the repository root:
```bash
bash scripts/convert_datasets.sh
```
This converts each task from raw data to LeRobot v2.1 and then to v3.0 (`datasets/lerobot/<Task>`).
`datasets_convert/v21_to_v30.py` is copied from LeRobot v0.5.0 and keeps the v2.1 copy as `datasets/lerobot/<Task>_old`. Delete both `datasets/lerobot/<Task>` and `datasets/lerobot/<Task>_old` before reconverting.

## Training

```bash
bash scripts/train_compvla.sh
```
The script reads `configs/config_compvla.yaml`; set `TASK` in the script to choose the dataset. Main options:

| Option | Description |
|---|---|
| `--policy.train_model_list` | Trainable flags for `[PaliGemma, action expert, compliance expert]` (`vel_expert`, `cp_expert` in the code) |
| `--policy.use_compliance_expert` | Use the Compliance Expert (`false` makes the model PI0 on velocity + gripper) |
| `--policy.use_joint_in_training` | Also predict joint positions as an auxiliary target |
| `--policy.traj_condition_space` | Trajectory condition of the Compliance Expert: `normalized` (default) integrates normalized twists, `physical` integrates physical-unit twists |
| `--policy.fps` | Must match the dataset fps (10 by default) |

RTC (Real-Time Chunking) is not supported with the Compliance Expert.

### GPU memory
For reference, peak memory measured on a single RTX 5090 (32 GB) with batch size 2, bf16, and gradient checkpointing:

| Setting | Trainable parameters | Peak memory |
|---|---|---|
| Full fine-tuning (`[true, true, true]`) | 4.08B | OOM on 32 GB |
| Frozen PaliGemma (`[false, true, true]`) | 1.16B | 13.5 GB |
| PEFT (LoRA) | 2.76M | 9.3 GB |

Memory depends on the batch size and settings. Full fine-tuning needs a GPU with more than 32 GB; multi-GPU training (data parallel) keeps a full copy of the model on each GPU, so it does not reduce the memory per GPU.
To train on multiple GPUs:
```bash
accelerate launch --multi_gpu --num_processes=2 train.py --config_path=configs/config_compvla.yaml [options]
```
If you hit CUDA OOM from memory fragmentation, uncomment the `PYTORCH_CUDA_ALLOC_CONF` line in the script.

### Resume
Point `--config_path` to the `train_config.json` of the checkpoint (under your `output_dir`):
```bash
python train.py --resume=true \
    --config_path=train_results/compvla/Erase/checkpoints/last/pretrained_model/train_config.json
```

### Training with PEFT (experimental)
Add `--peft.method_type=LORA` (optionally `--peft.r=...`). PEFT trains far fewer parameters, so we use a 10x larger LR than full fine-tuning:
```bash
--policy.optimizer_lr=2.5e-4 --policy.scheduler_decay_lr=2.5e-5
```

## License
The code is released under the Apache License 2.0 (see `LICENSE`).
Parts of the code are adapted from [LeRobot](https://github.com/huggingface/lerobot) and [OpenPI](https://github.com/Physical-Intelligence/openpi), both licensed under the Apache License 2.0; those files keep their original copyright notices.

The dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation
```bibtex
@article{kim2026compvla,
  title={CompVLA: A Variable Compliance Vision-Language-Action Model for Contact-rich Manipulation},
  author={Kim, Jongmin and Ha, Junsu and Park, Che-Sang and Song, Minchang and Jeong, Hyeokju and Hwang, Himchan and Fu, Jianlong and Park, Frank C},
  journal={arXiv preprint arXiv:2609.23614},
  year={2026}
}
```

## Acknowledgements
This code builds on [LeRobot](https://github.com/huggingface/lerobot) and [OpenPI](https://github.com/Physical-Intelligence/openpi).
