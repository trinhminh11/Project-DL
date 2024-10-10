# Standard library imports
import random
import os
from abc import ABC, abstractmethod
from typing import Type, Callable, Any

# Third-party imports
import gymnasium as gym
import numpy as np
from numpy._typing import _ShapeLike
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from .replay_buffer import ReplayBuffer

from .agent_callbacks import Callbacks


class AgentBase(ABC):
    def __init__(
            self,
            state_shape: _ShapeLike,
            action_shape: _ShapeLike,
            batch_size: int,
            on_policy: bool,
            update_every: int = 1,
            buffer_size: int = int(1e5),
            device: str = 'cpu',
            seed = 0,
            **kwargs
        ) -> None:
        
        self.device = device
        self.on_policy = on_policy
        self.update_every = update_every
        self.memory = ReplayBuffer(state_shape, action_shape, batch_size, on_policy, buffer_size, device, seed)
        self.time_step = 0
        self.eval = False
        random.seed(seed)

        self._modules: dict[str, nn.Module] = {}

        self.callbacks = Callbacks(self)

        self.epoch = None

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, nn.Module):
            self._modules[name] = value

        super().__setattr__(name, value)
    
    def apply(self, fn: Callable[[nn.Module], None]):
        for module in self._modules.values():
            module.apply(fn)
    
    def to(self, device):
        self.device = device
        self.memory.to(device)

        for module in self._modules.values():
            module.to(device)

    def save(self, dir):
        dir = os.path.join(dir, self.__class__.__name__)

        if not os.path.exists(dir):
            os.makedirs(dir)

        for name, module in self._modules.items():
            torch.save(module.state_dict(), os.path.join(dir, name + ".pth"))
    
    def load(self, dir = None):
        for name, module in self._modules.items():
            module.load_state_dict(torch.load(os.path.join(dir, self.__class__.__name__, name + ".pth"), self.device, weights_only=True))

    def reset(self):
        r"""
        This method should call at the start of an episode (after :func:``env.reset``)
        """
        ...
    
    @abstractmethod
    def act(self, state: np.ndarray) -> Any: 
        """
        Abstract method to be implemented by subclasses to define the action 
        taken by the agent given a certain state.

        Args:
            state (np.ndarray): The current state represented as a NumPy array.

        Returns:
            Any: The action to be taken by the agent. The type of the action 
            depends on the specific implementation.
        """
        ...

    @abstractmethod
    def learn(self, states: Tensor, actions: Tensor, rewards: Tensor, next_states: Tensor, terminals: Tensor) -> Any: 
        """
        Abstract method to be implemented by subclasses for learning from a batch of experiences.

        Note:
            - If the agent is on-policy, this method should be called after each episode.
            - If the agent is off-policy, this method should be called after each step.

        Args:
            states (Tensor): The batch of current states.
            actions (Tensor): The batch of actions taken.
            rewards (Tensor): The batch of rewards received.
            next_states (Tensor): The batch of next states resulting from the actions.
            terminals (Tensor): The batch of terminal flags indicating if the next state is terminal.

        """
        ...
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, terminal: np.ndarray):
        """
        Perform a single step in the agent's interaction with the environment.

        This method adds the current experience to the agent's memory and, if the agent is off-policy,
        periodically samples a batch from the memory to learn from it.

        Args:
            state (object): The current state of the environment.
            action (object): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (object): The state of the environment after taking the action.
            terminal (bool): Whether the episode has ended.

        Returns:
            None
        """
        # Add the current experience to the replay buffer
        self.memory.add(state, action, reward, next_state, terminal)

        # If the agent is on-policy, return immediately
        if self.on_policy:
            return

        # Increment the time step and check if it's time to update
        self.time_step = (self.time_step + 1) % self.update_every

        # If it's time to update, sample a batch from memory and learn from it
        if self.time_step == 0:
            if len(self.memory) >= self.memory.batch_size:
                states, actions, rewards, next_states, terminals = self.memory.sample()
                self.learn(states, actions, rewards, next_states, terminals)

    def train_on_episode(self, env: gym.Env, max_t: int, callbacks: Type[Callbacks] = None) -> float:
        """
        Train the agent on a single episode.
        Args:
            env (gym.Env): The environment to train the agent in.
            max_t (int): The maximum number of time steps in the episode.
            progress_bar (Type[tqdm], optional): Progress bar for visualizing the training process. Defaults to None.
            callbacks (Type[Callbacks], optional): Callbacks for custom actions at different stages of training. Defaults to None.
        Returns:
            float: The total score (sum of rewards) obtained in the episode.
        """
        if callbacks is None:
            callbacks = Callbacks(self)
        
        callbacks.on_episode_begin()

        self.eval = False

        obs = env.reset()[0]
        self.reset()
        score = 0

        for time_step in range(max_t):
            callbacks.on_step_begin()
            action = self.act(obs)
            next_obs, reward, terminal, truncated, info = env.step(action)

            done = terminal or truncated

            self.step(obs, action, reward, next_obs, terminal)
            obs = next_obs

            score += reward

            callbacks.on_step_end()

            if done:
                break
        
        if self.on_policy:
            states, actions, rewards, next_states, terminals = self.memory.sample()
            self.memory.clear()
            
            self.learn(states, actions, rewards, next_states, terminals)

        callbacks.on_episode_end()
        
        return score


    def fit(self, env: gym.Env, n_games: int, max_t: int, save_best=False, save_last=False, save_dir="./", progress_bar: Type[tqdm] = None, callbacks: Type[Callbacks] = None) -> list:
        """
        Train the agent on the given environment for a specified number of games.
        Args:
            env (gym.Env): The environment to train the agent on.
            n_games (int): Number of games to train the agent.
            max_t (int): Maximum number of timesteps per game.
            save_best (bool, optional): If True, save the model when it achieves the best score. Defaults to False.
            save_last (bool, optional): If True, save the model after the last game. Defaults to False.
            save_dir (str, optional): Directory to save the model. Defaults to "./".
            progress_bar (Type[tqdm], optional): Progress bar for visualizing training progress. Defaults to None.
            callbacks (Type[Callbacks], optional): Callbacks to execute during training. Defaults to None.
        Returns:
            list: A list of scores for each game played.
        """
        if callbacks is None:
            callbacks = Callbacks(self)
        
        callbacks.on_train_begin()

        scores = []

        # Use tqdm for progress bar if provided
        loop = progress_bar(range(n_games)) if progress_bar else range(n_games)

        for self.epoch in loop:
            score = self.train_on_episode(env, max_t, callbacks)
            
            scores.append(score)

            avg_score = np.mean(scores[-100:])

            if save_best:
                self.save(save_dir)

            else:
                if save_last:
                    self.save(save_dir)

            if progress_bar:
                loop.set_postfix(score = score, avg_score = avg_score)
            
        callbacks.on_train_end()
        
        self.epoch = None
        return scores

    def play(self, env: gym.Env):
        self.eval = True
        import pygame
        
        pygame.init()
        score = 0
        obs = env.reset()[0]
        self.reset()

        done = False
        while not done:
            env.render()
            action = self.act(obs)

            next_obs, reward, done, truncated, info = env.step(action)

            obs = next_obs

            score += reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        pygame.quit()

        self.eval = False

        return score
