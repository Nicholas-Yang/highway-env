import gym
import highway_env
from stable_baselines3 import DQN
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

models_dir = "D:/2022 FALL HOMEWORK/AUE8930/HW5/models.zip"

env = gym.make("highway-fast-v0") 

model = DQN.load(models_dir, env=env)

episodes = 100

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, done, info = env.step(int(action))
        env.render()
        print(rewards)