from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor,
                                              VecVideoRecorder)
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, FlatObsWrapper, RGBImgObsWrapper, FullyObsWrapper, OneHotPartialObsWrapper
from UCB0.ddqn import DQN
# from stable_baselines3 import DQN
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        # print(observation_space.shape)
        # print('hi')
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.last_mean = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 2000 timesteps
              mean_reward = np.mean(y[self.last_mean:])
              self.last_mean = len(y)
              result_list.append(mean_reward)
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            #   # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
            #       # Example for saving best model
            #       if self.verbose >= 1:
            #         print(f"Saving new best model to {self.save_path}")
            #       self.model.save(self.save_path)

        return True
env_names = ["MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0", "MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-MultiRoom-N2-S4-v0"]
for env_name in env_names:
    alg_name = "UCB0"
    log_dir = "tmp/"
    timesteps = 1e5
    result_list = [] 
    result_list.append(alg_name)

    if not os.path.exists('result_'+env_name+'.csv'):
        columns = []
        columns.append('ALGO')
        for i in range(50):
            columns.append(i*2000+2000)
        if columns is None:
            raise ValueError("새 CSV 파일을 만들기 위해서는 'columns' 매개변수가 필요합니다.")
        df = pd.DataFrame(columns=columns)
        # 새 DataFrame을 CSV 파일로 저장
        df.to_csv('result_'+env_name+'.csv', index=False)

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    obs = env.reset()
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    env = model.get_env()
    # env = VecVideoRecorder(env, alg_name+'_'+env_name, record_video_trigger=lambda x: x % 1e4 == 0, video_length=200)
    env = VecMonitor(env, log_dir)


    obs = env.reset()
    # print(env.num_envs)
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(check_freq=2e3, log_dir=log_dir)
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(timesteps, callback=callback)
    model.save(os.path.join(log_dir, alg_name+'_'+str(int(timesteps))+'_'+env_name))
    data = pd.read_csv('result_'+env_name+'.csv')
    # data = data.append(pd.Series(result_list, index=data.columns[:len(result_list)]), ignore_index=True)
    data.loc[len(data.index)] = pd.Series(result_list, index=data.columns[:len(result_list)])
    data.to_csv('result_'+env_name+'.csv',index=False)