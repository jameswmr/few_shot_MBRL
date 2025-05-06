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

