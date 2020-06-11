#!/bin/bash
python3 -m venv algaedice_env
source algaedice_env/bin/activate
pip install --upgrade pip
pip install tensorflow==2.0.0
pip install tensorboard==2.0.1
pip install tensorflow-probability==0.8.0
pip install tf-agents-nightly==0.2.0.dev20191125
pip install tfp-nightly==0.9.0.dev20191125
pip install tqdm>=4.36.1
pip install gast==0.2.2
pip install networkx
pip install dotmap
pip install pygame
pip install gym
pip install mujoco_py
pip install -e ~/code/d4rl
