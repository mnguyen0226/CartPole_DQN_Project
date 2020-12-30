import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

# # Check environment wise, no training yet ########################
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(100):
#     print("Running")
#     env.render()
#     env.step(env.action_space.sample())
#     print("Finish")
#
# env.close()

env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1', force=True)
observation = env.reset()
for t in range(100):
#    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.monitor.close()