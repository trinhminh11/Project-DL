# Standard library imports
import random
import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Callable, Any

# Third-party imports
import gymnasium as gym
import numpy as np
from numpy._typing import _ShapeLike
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from .buffers import ReplayBuffer, RolloutBuffer, BaseBuffer

from .agent_callbacks import Callbacks

from .. import utils
from .. import core

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class AgentBase(ABC):
    memory: BaseBuffer
    def __init__(
            self,
            policy: nn.Module,
            env: core.EnvWithTransform | gym.Env | str,
            optimizer: Type[optim.Optimizer] = optim.Adam,
            lr: float = 1e-3,
            optimizer_kwargs: dict = None,
            device: str = 'auto',
            seed = None,
        ):
        if not isinstance(policy, nn.Module):
            raise ValueError("policy must be an instance of torch.nn.Module")

        if not issubclass(optimizer, optim.Optimizer):
            raise ValueError("optimizer must be a subclass of torch.optim.Optimizer")

        if isinstance(env, gym.vector.VectorEnv):
            if isinstance(env, core.EnvWithTransform):
                self.env = env
            else:
                self.env = core.EnvWithTransform(env)
        else:
            if isinstance(env, str):
                self.env = core.make_vec(env, 1)
            elif isinstance(env, core.EnvWithTransform):
                self.env = core.EnvWithTransform(gym.vector.AsyncVectorEnv([lambda : env.env]))
            elif isinstance(env, gym.Env):
                self.env = core.EnvWithTransform(gym.vector.AsyncVectorEnv([lambda : env.env]))
            else:
                raise ValueError("env must be an instance of EnvWithTransform, gym.Env, or str")

        self.n_envs = self.env.unwrapped.num_envs


        self.policy = policy

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.optimizer = optimizer(self.policy.parameters(), lr = lr, **optimizer_kwargs)
        
        self.device = utils.get_device(device)
        self.seed = seed

        self.memory = None

        self.to(self.device)
    
    def apply(self, fn: Callable[[nn.Module], None]):
        self.policy.apply(fn)
    
    def to(self, device):
        self.device = device
        if self.memory:
            self.memory.to(device)
        self.policy.to(device)
        
        return self

    def save(self, dir, *post_names):
        name = self.__class__.__name__

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        for post_name in post_names:
            name += "_" + post_name
        
        torch.save(self.policy.state_dict(), os.path.join(dir, name + ".pth"))

    
    def load(self, dir, *post_names):
        name = self.__class__.__name__
        for post_name in post_names:
            name += "_" + post_name

        self.policy.load_state_dict(torch.load(os.path.join(dir, name + ".pth"), self.device, weights_only=True))

    def reset(self):
        r"""
        This method should call at the start of an episode (after :func:``env.reset``)
        """
        ...
    
    @abstractmethod
    def act(self, observation: np.ndarray | dict, deterministic: bool = True) -> ActType:
        r"""
        Abstract method to be implemented by subclasses to define the action 
        taken by the agent given a certain observation.

        Args:
            observation (np.ndarray): The current observation represented as a NumPy array.

        Returns:
            ActType: The action to be taken by the agent.
        """
        ...
        raise NotImplementedError
    
    def _act(self, observation: np.ndarray | dict, deterministic: bool = True) -> ActType:
        actions = []
        if isinstance(observation, dict):
            _observations = [{}] * self.n_envs

            for i in range(self.n_envs):
                for key, obs in observation.items():
                    _observations[i][key] = obs[i]
                actions.append(self.act(_observations[i], deterministic))

        else:
            for i in range(self.n_envs):
                actions.append(self.act(observation[i], deterministic))
        
        return actions

    
    def play(self, env: core.EnvWithTransform, stop_if_truncated: bool = False, seed = None) -> float:
        self.eval = True
        import pygame
        
        pygame.init()
        score = 0
        obs = env.reset(seed=seed)[0]
        self.reset()

        done = False
        while not done:
            env.render()
            action = self.act(obs, True)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated

            if stop_if_truncated:
                done = done or truncated

            obs = next_obs

            score += env.env_reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        pygame.quit()

        self.eval = False

        return score
    
    def play_jupyter(self, env: core.EnvWithTransform, stop_if_truncated: bool = False, seed = None) -> float:
        self.eval = True
        import pygame
        from IPython.display import display
        from PIL import Image

        pygame.init()
        score = 0
        obs = env.reset(seed=seed)[0]
        self.reset()

        done = False
        while not done:
            env.render()
            action = self.act(obs, True)

            next_obs, reward, terminated, truncated, info = env.step(action)

            pixel = env.render()

            display(Image.fromarray(pixel), clear=True)

            done = terminated

            if stop_if_truncated:
                done = done or truncated

            obs = next_obs

            score += env.env_reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        pygame.quit()

class OffPolicyAgent(AgentBase):
    memory: ReplayBuffer
    def __init__(
            self, 
            policy: nn.Module, 
            env: core.EnvWithTransform | gym.Env | str, 
            optimizer: Type[optim.Optimizer] = optim.Adam,
            lr: float = 1e-3,
            optimizer_kwargs: dict = None,
            buffer_size = int(1e5),
            batch_size: int = 64,
            update_every: int = 1,
            device = 'auto', 
            seed = None
        ):
        super().__init__(policy, env, optimizer, lr, optimizer_kwargs, device, seed)

        self.memory = ReplayBuffer(buffer_size = buffer_size, observation_space = self.env.observation_space, action_space = self.env.action_space, batch_size = batch_size, device = self.device, n_envs = self.n_envs, seed = self.seed)

        self.update_every = update_every

        self.time_step = 0

    
    @abstractmethod
    def learn(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> None:
        """
        Abstract method to be implemented by subclasses for learning from a batch of experiences.

        Args:
            observations (torch.Tensor): The batch of current observations.
            actions (torch.Tensor): The batch of actions taken.
            rewards (torch.Tensor): The batch of rewards received.
            next_observations (torch.Tensor): The batch of next observations resulting from the actions.
            terminals (torch.Tensor): The batch of terminal flags indicating if the next observation is terminal.
        """
        ...
    
    def step(self, observation: np.ndarray | dict, action: np.ndarray | dict, reward: np.ndarray, next_observation: np.ndarray | dict, terminal: np.ndarray):
        """
        Perform a single step in the agent's environment.
        This method adds the current experience to the replay buffer, increments the time step,
        and updates the agent by sampling a batch from memory if it's time to update.
        Args:
            observation (np.ndarray): The current state observation.
            action (np.ndarray): The action taken by the agent.
            reward (np.ndarray): The reward received after taking the action.
            next_observation (np.ndarray): The next state observation after taking the action.
            terminal (np.ndarray): A flag indicating whether the episode has ended.
        Returns:
            None
        """
        
        # Add the current experience to the replay buffer
        self.memory.add(observation, action, reward, next_observation, terminal)

        # Increment the time step and check if it's time to update
        self.time_step = (self.time_step + 1) % self.update_every

        # If it's time to update, sample a batch from memory and learn from it
        if self.time_step == 0:
            if len(self.memory) >= self.memory.batch_size:
                observations, actions, rewards, next_observations, terminals = self.memory.sample()
                for i in range(self.n_envs):
                    self.learn(observations[i], actions[i], rewards[i], next_observations[i], terminals[i])


    def fit(self, total_timesteps: int = None, n_games: int = None, deterministic=False, save_best=False, save_every=False, save_dir="./", progress_bar: Type[tqdm] = None, callbacks: Type[Callbacks] = None) -> list:
        if callbacks is None:
            callbacks = Callbacks(self)

        if total_timesteps is None and n_games is None:
            raise ValueError("Either total_timesteps or n_games must be provided")

        if total_timesteps:
            return self.fit_by_timesteps(total_timesteps, deterministic, save_best, save_every, save_dir, progress_bar, callbacks)
        else:
            return self.fit_by_games(n_games, deterministic, save_best, save_every, save_dir, progress_bar, callbacks)

    def fit_by_timesteps(self, total_timesteps: int, deterministic = False, save_best = False, save_every = False, save_dir = "./", progress_bar: Type[tqdm] = None, callbacks: Type[Callbacks] = None):
        if callbacks is None:
            callbacks = Callbacks(self)

        callbacks.on_train_begin()

        scores = []
        mean_scores = []

        # Use tqdm for progress bar if provided
        loop: tqdm = progress_bar(range(total_timesteps)) if progress_bar else range(total_timesteps)

        episode = 0
        
        start = True

        best_score = float('-inf')

        for t in loop:
            if start:
                score = 0
                callbacks.on_episode_begin()
                obs = self.env.reset()[0]
                self.reset()
                start = False
            
            action = self._act(obs, deterministic)
            next_obs, reward, terminal, truncated, info = self.env.step(action)
            done = (terminal or truncated).all()
            self.step(obs, action, reward, next_obs, terminal)
            score += self.env.env_reward.mean()
            obs = next_obs

            if done:
                callbacks.on_episode_end()
                start = True
                scores.append(score)
                episode += 1

                avg_score = np.mean(scores[-100:])
                mean_scores.append(avg_score)

                if save_best:
                    if avg_score >= best_score:
                        best_score = avg_score
                        self.save(save_dir, "best")
                
                if save_every:
                    if episode % save_every == 0:
                        self.save(save_dir, str(episode))
            
                if progress_bar:
                    loop.set_postfix({"game": episode, "avg_score": np.mean(scores[-100:]), "score": scores[-1]})

        callbacks.on_train_end()
        return {'scores': scores, 'mean_scores': mean_scores, 'n_games': episode, 'total_timesteps': total_timesteps}

    def fit_by_games(self, n_games: int, deterministic = False, save_best = False, save_every = False, save_dir = "./", progress_bar: Type[tqdm] = None, callbacks: Type[Callbacks] = None):
        if callbacks is None:
            callbacks = Callbacks(self)
        
        callbacks.on_train_begin()

        scores = []
        mean_scores = []

        # Use tqdm for progress bar if provided
        loop = progress_bar(range(n_games)) if progress_bar else range(n_games)

        total_timesteps = 0

        best_score = float('-inf')

        for episode in loop:
            done = False
            score = 0
            callbacks.on_episode_begin()
            obs = self.env.reset()[0]
            self.reset()
            while not done:
                total_timesteps += 1
                action = self._act(obs, deterministic)
                next_obs, reward, terminal, truncated, info = self.env.step(action)

                done = (terminal or truncated).all()
                self.step(obs, action, reward, next_obs, terminal)
                score += self.env.env_reward.mean()
                obs = next_obs

            callbacks.on_episode_end()

            scores.append(score)
            avg_score = np.mean(scores[-100:])
            mean_scores.append(avg_score)

            if save_best:
                if avg_score >= best_score:
                    best_score = avg_score
                    self.save(save_dir, "best")
            
            if save_every:
                if episode % save_every == 0:
                    self.save(save_dir, str(episode))

            if progress_bar:
                loop.set_postfix({"avg_score": avg_score, "score": scores[-1]})

        callbacks.on_train_end()

        return {'scores': scores, 'mean_scores': mean_scores, 'n_games': n_games, 'total_timesteps': total_timesteps}