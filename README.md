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
# How to run
## Data collection for dynamic model
```
python scripts/data_collection.py --n_episodes 10000 --seed 0 --output_dir data.pt
```
## Data collection for encoder
```
python scripts/data_collection_encoder.py --n_episodes 1000 --seed 0 --dym_dir best_rnn_32.pt --output_dir data_encoder.pt
```

## Train dynamic model
```
python scripts/train_dynamic.py --seed 0 --method rnn --data_dir data.pt --output_dir best_dynamic.pt 
```
## Train encoder
```
python scripts/train_encoder.py --seed 0 --data_dir data_encoder.pt --shot-steps 10 --dynamic-model-ckpt best_rnn_32.pt 
```
## Train RL
```
python scripts/train_task_policy.py --seed 0 --policy_name ppo --dr False
```
## Eval dynamic model
```
python scripts/eval_dynamic.py --seed 0 --dym_dir best_rnn_32.pt --data_dir data_eval.pt --sample_train True --n_episodes 100 --method rnn 
```

## Eval 
```
python scripts/eval.py --seed 0 --method rnn --dym_dir best_rnn_32.pt --enc_dir best_encoder.pt --sample_train True --data_dir data_eval.pt 
```

## Eval RL
```
python scripts/eval_rl.py --seed 0 --policy_name ppo --sample_train True --dym_dir ppo --data_dir data_eval.pt
```

## Run ood test
Change gym_envs/utilities lines 46,47 "dynamic@floor1_table_collision@friction_sliding": [0.02, 0.07] to "dynamic@floor1_table_collision@friction_sliding": [0.02, 0.2] 


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
python few_shot_MBRL/baseline2/eval.py \
    --am_model outputs/baseline2_stage_2/adaptation_module.pth \
    --base_policy_model output/stage1_sb3/ckpt_best.zip \
    --ood
```

# Download our model
```
gdown https://drive.google.com/drive/folders/1h5T23qQotc6F39r5wN_4NY-UUmspsiPU?usp=drive_link
```
Make sure model is under output/ and data is under data/
