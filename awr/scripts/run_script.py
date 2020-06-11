import argparse
import gym
import uuid
import numpy as np
import json
import os
import sys
import tensorflow as tf

import d4rl

import awr_configs
import learning.awr_agent as awr_agent

arg_parser = None

def parse_args(args):
    parser = argparse.ArgumentParser(description="Train or test control policies.")

    parser.add_argument("--env", dest="env", default="")

    parser.add_argument("--train", dest="train", action="store_true", default=True)
    parser.add_argument("--test", dest="train", action="store_false", default=True)

    parser.add_argument("--max_iter", dest="max_iter", type=int, default=np.inf)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_episodes", dest="test_episodes", type=int, default=32)
    parser.add_argument("--output_dir", dest="output_dir", default="output")
    parser.add_argument("--output_iters", dest="output_iters", type=int, default=50)
    parser.add_argument("--model_file", dest="model_file", default="")

    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--gpu", dest="gpu", default="")

    arg_parser = parser.parse_args()

    return arg_parser

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

def build_env(env_id):
    assert(env_id is not ""), "Unspecified environment."
    env = gym.make(env_id)
    return env

def build_agent(env):
    env_id = arg_parser.env
    agent_configs = {}
    if (env_id in awr_configs.AWR_CONFIGS):
        agent_configs = awr_configs.AWR_CONFIGS[env_id]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

    return agent

def main(args):
    d4rl.set_dataset_path('/datasets')

    global arg_parser
    arg_parser = parse_args(args)
    enable_gpus(arg_parser.gpu)

    # Setup logging
    final_output_dir = os.path.join(arg_parser.output_dir, str(uuid.uuid4()))
    os.makedirs(final_output_dir, exist_ok=True)
    with open(os.path.join(final_output_dir, 'params.json'), 'w') as params_file:
        json.dump({
            'env_name': arg_parser.env,
            'seed': arg_parser.seed,
        }, params_file)

    env = build_env(arg_parser.env)

    agent = build_agent(env)
    agent.visualize = arg_parser.visualize
    if (arg_parser.model_file is not ""):
        agent.load_model(arg_parser.model_file)


    if (arg_parser.train):
        agent.train(max_iter=arg_parser.max_iter,
                    test_episodes=arg_parser.test_episodes,
                    output_dir=final_output_dir,
                    output_iters=arg_parser.output_iters)
    else:
        agent.eval(num_episodes=arg_parser.test_episodes)

    return

if __name__ == "__main__":
    main(sys.argv)
