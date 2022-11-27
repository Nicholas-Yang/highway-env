import gym
import highway_env
from stable_baselines3 import DQN
import os
from torch.utils.tensorboard import SummaryWriter



models_dir = "D:/2022 FALL HOMEWORK/AUE8930/HW5/models"
logdir = "highway_env"
writer = SummaryWriter(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


env = gym.make("highway-fast-v0")
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log=logdir)
model.learn(int(2e4))
model.save(models_dir)

model = DQN.load(models_dir)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()