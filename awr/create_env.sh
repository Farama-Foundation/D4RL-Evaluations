#!/bin/bash
python3 -m venv awr_env
source awr_env/bin/activate
pip install --upgrade pip
pip install tensorflow==1.14.0
pip install networkx
pip install dotmap
pip install pygame
pip install gym
pip install mujoco_py
pip install -e ~/code/d4rl
