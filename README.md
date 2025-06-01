# Installation
```
conda env create -f environment.yaml
conda activate EV_DRL
cd ./gym_env
pip install -e .
cd ../
python gym_env/env_push_one.py
```

## Install with uv

Tested on macOS.

```
uv venv --python 3.8
uv pip install matplotlib imageio imageio-ffmpeg gym tqdm wandb opencv-python termcolor mujoco numba scipy
uv pip install 'pyobjc-core<11.0' 'pyobjc-framework-Quartz<11.0'
cd gym_env
uv pip install --no-deps -e .
```

Run to generate new `dummyDemo_video.mp4` and `cup_trajectory.png` files.

```
python gym_env/env_push_one.py
```

# Run baseline
Check scripts/train_task_policy.py for example

# Run baseline2

## Train

```
# Stage 1
python -m few_shot_MBRL.baseline2.baseline2_stage_1 \
    --device cuda:0 --total_timesteps 1_000_000 --n_envs 16 --headless --wandb

# Stage 2
python -m few_shot_MBRL.baseline2.baseline2_stage_2 \
    --training_set_size 10000 \
    --output_dir outputs/baseline2_stage_2 \
    --num_workers 10
```

## Evaluate

```
python -m few_shot_MBRL.baseline2.eval \
    --am_model outputs/baseline2_stage_2/adaptation_module.pth \
    --base_policy_model output/stage1_sb3/ckpt_best.zip
```
