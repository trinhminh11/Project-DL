# Standard library imports
import random
import os
from abc import ABC, abstractmethod
from typing import Type, Any

# Third-party imports
import gymnasium as gym
import numpy as np
from numpy._typing import _ShapeLike
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_shape: _ShapeLike, action_shape: _ShapeLike, buffer_size: int, batch_size: int, device: str, seed = 0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_shape (int): dimension of each state
            action_shape (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.mem_size = buffer_size

        self.state_memory = np.zeros([buffer_size, *state_shape], dtype=np.float32)
        self.action_memory = np.zeros([buffer_size, *action_shape], dtype=np.float32)
        self.reward_memory = np.zeros([buffer_size, 1], dtype=np.float32)
        self.next_state_memory = np.zeros([buffer_size, *state_shape], dtype=np.float32)
        self.terminal_memory = np.zeros([buffer_size, 1], dtype=np.bool_)

        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

        self.mem_cntr = 0

    def to(self, device):
        self.device = device

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, terminal: np.ndarray):
        """Add a new experience to memory."""
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = terminal

        self.mem_cntr += 1

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(range(len(self)), self.batch_size)

        states = torch.from_numpy(self.state_memory[batch]).to(self.device)
        actions = torch.from_numpy(self.action_memory[batch]).to(self.device)
        rewards = torch.from_numpy(self.reward_memory[batch]).to(self.device)
        next_states = torch.from_numpy(self.next_state_memory[batch]).to(self.device)
        terminals = torch.from_numpy(self.terminal_memory[batch]).bool().to(self.device)

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        """Return the current size of internal memory."""
        return min(self.mem_cntr, self.mem_size)


class AgentBase(ABC):
    def __init__(
            self,
            state_shape: _ShapeLike,
            action_shape: _ShapeLike,
            batch_size: int,
            max_mem_size: int,
            update_every: int,
            device: str,
            seed,
            **kwargs
        ) -> None:
        
        self.device = device
        self.update_every = update_every
        self.memory = ReplayBuffer(state_shape, action_shape, max_mem_size, batch_size, device, seed)
        self.time_step = 0
        self.eval = False
        random.seed(seed)

        self._modules: dict[str, nn.Module] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, nn.Module):
            self._modules[name] = value

        super().__setattr__(name, value)

    def step(self, state, action, reward, next_state, terminal):
        """
        Save the experience (state, action, reward, next_state, terminal) to the replay buffer
        and call the learn method every 'update_every' steps.
        """

        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, terminal)

        # Increment time step and check if it's time to learn
        self.time_step = (self.time_step + 1) % self.update_every

        if self.time_step == 0:
            # Learn if there are enough samples in the replay buffer
            if len(self.memory) >= self.memory.batch_size:
                states, actions, rewards, next_states, terminals = self.memory.sample()
                self.learn(states, actions, rewards, next_states, terminals)

    def reset(self):
        r"""
        This method should call at the start of an episode (after :func:``env.reset``)
        """
        ...

    @abstractmethod
    def act(self, state: np.ndarray) -> Any: ...

    @abstractmethod
    def learn(self, states: Tensor, actions: Tensor, rewards: Tensor, next_states: Tensor, terminals: Tensor) -> Any: ...

    def save(self, dir):
        dir = os.path.join(dir, self.__class__.__name__)

        if not os.path.exists(dir):
            os.makedirs(dir)

        for name, module in self._modules.items():
            torch.save(module.state_dict(), os.path.join(dir, name + ".pth"))
    
    def load(self, dir):
        for name, module in self._modules.items():
            module.load_state_dict(torch.load(os.path.join(dir, name + ".pth"), self.device, weights_only=True))

    def fit(self, env: gym.Env, n_games: int, max_t: int, save_best=False, save_last=False, save_dir="./", progress_bar: Type[tqdm] = None):
        self.eval = False
        scores = []

        # Use tqdm for progress bar if provided
        loop = progress_bar(range(n_games)) if progress_bar else range(n_games)

        for episode in loop:
            score = 0
            obs = env.reset()[0]
            self.reset()

            for time_step in range(max_t):
                action = self.act(obs)
                next_obs, reward, terminal, truncated, info = env.step(action)

                self.step(obs, action, reward, next_obs, terminal)
                obs = next_obs

                score += reward

                if terminal:
                    break
            
            scores.append(score)

            avg_score = np.mean(scores[-100:])

            if save_best:
                self.save(save_dir)

            else:
                if save_last:
                    self.save(save_dir)

            if progress_bar:
                loop.set_postfix(score = score, avg_score = avg_score)
        
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
