from typing import Literal
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import Tensor
import numpy as np
from numpy._typing import _ShapeLike
import random
import gymnasium.spaces as spaces

from .. import utils


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        n_envs: int = 1,
        seed: int = None,
    ):
        if n_envs <= 0:
            raise ValueError("The number of environments must be greater than 0.")
        
        utils.check_for_nested_spaces(observation_space)
        utils.check_for_nested_spaces(action_space)
        
        self.buffer_size = buffer_size
        self.obs_shape = utils.get_shape(observation_space)[1:]
        self.action_shape = utils.get_shape(action_space)[1:]
        self.n_envs = n_envs
        self.device = utils.get_device(device)

        self.mem_cntr = 0
        self.full = False

        self.seed = random.seed(seed)
    
    def to(self, device: torch.device | str) -> None:
        self.device = utils.get_device(device)
    
    def __len__(self):
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.mem_cntr

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def sample(self):
        """
        Sample elements from the buffer.
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """
        Reset the buffer.
        """
        self.mem_cntr = 0
        self.full = False

class ReplayBuffer(BaseBuffer):

    observations: np.ndarray | dict[str, np.ndarray]
    actions: np.ndarray | dict[str, np.ndarray]
    rewards: np.ndarray
    next_observations: np.ndarray | dict[str, np.ndarray]
    terminals: np.ndarray

    def __init__(
            self, 
            buffer_size: int, 
            observation_space: spaces.Space, 
            action_space: spaces.Space, 
            batch_size: int, 
            device = "auto", 
            n_envs: int = 1, 
            seed: int = None
        ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs, seed)


        if isinstance(self.obs_shape, dict):
            self.observations = {key: np.zeros([self.buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.obs_shape.items()}
        else:
            self.observations = np.zeros([buffer_size, n_envs, *self.obs_shape], dtype=np.float32)
        
        if isinstance(self.action_shape, dict):
            self.actions = {key: np.zeros([self.buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.action_shape.items()}
        else:
            self.actions = np.zeros([buffer_size, n_envs, *self.action_shape], dtype=np.float32)
        
        self.rewards = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.next_observations = deepcopy(self.observations)

        self.terminals = np.zeros([buffer_size, n_envs], dtype=np.bool_)

        self.batch_size = batch_size

    def add(
            self, 
            observation: np.ndarray | dict[str, np.ndarray], 
            action: np.ndarray | dict[str, np.ndarray], 
            reward: np.ndarray, 
            next_observation: np.ndarray | dict[str, np.ndarray], 
            terminal: np.ndarray
        ) -> None:
        """Add a new experience to memory."""
        idx = self.mem_cntr

        if isinstance(self.observations, dict):
            for key in observation.keys():
                self.observations[key][idx] = observation[key]
                self.next_observations[key][idx] = next_observation[key]
        else:
            self.observations[idx] = observation
            self.next_observations[idx] = next_observation
        
        if isinstance(self.actions, dict):
            for key in action.keys():
                self.actions[key][idx] = action[key]
        else:
            self.actions[idx] = action

        self.rewards[idx] = reward
        self.terminals[idx] = terminal

        self.mem_cntr += 1

        if self.mem_cntr == self.buffer_size:
            self.full = True
            self.mem_cntr = 0

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample a batch of experiences from memory."""
        batch = np.zeros((self.n_envs, self.batch_size), dtype=int)
        for i in range(self.n_envs):
            batch[i] = random.sample(range(len(self)), self.batch_size)

        return self._get_batch(batch)
    
    def _get_batch(self, batch: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if isinstance(self.obs_shape, dict):
            observations = [{}]*self.n_envs
            next_observations = [{}]*self.n_envs
        else:
            observations = torch.zeros((self.n_envs, self.batch_size, *self.obs_shape)).float().to(self.device)
            next_observations = torch.zeros((self.n_envs, self.batch_size, *self.obs_shape)).float().to(self.device)

        if isinstance(self.action_shape, dict):
            actions = [{}]*self.n_envs
        else:
            actions = torch.zeros((self.n_envs, self.batch_size, *self.action_shape)).float().to(self.device)
        
        rewards = torch.zeros((self.n_envs, self.batch_size)).float().to(self.device)
        terminals = torch.zeros((self.n_envs, self.batch_size)).bool().to(self.device)

        for i in range(self.n_envs):
            if isinstance(self.observations, dict):
                observations[i] = {key: torch.from_numpy(self.observations[key][batch[i], i]).to(self.device) for key in self.observations.keys()}
                next_observations[i] = {key: torch.from_numpy(self.next_observations[key][batch[i], i]).to(self.device) for key in self.next_observations.keys()}
            else:
                observations[i] = torch.from_numpy(self.observations[batch[i], i]).to(self.device)
                next_observations[i] = torch.from_numpy(self.next_observations[batch[i], i]).to(self.device)

            if isinstance(self.actions, dict):
                actions[i] = {key: torch.from_numpy(self.actions[key][batch[i], i]).to(self.device) for key in self.actions.keys()}
            else:
                actions[i] = torch.from_numpy(self.actions[batch[i], i]).to(self.device)

            rewards[i] = torch.from_numpy(self.rewards[batch[i], i]).to(self.device)
            terminals[i] = torch.from_numpy(self.terminals[batch[i], i]).bool().to(self.device)
        
        return observations, actions, rewards, next_observations, terminals

class RolloutBuffer(BaseBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray

    advantages: np.ndarray
    returns: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
            self, 
            buffer_size: int, 
            observation_space: spaces.Space, 
            action_space: spaces.Space, 
            batch_size: int | None, 
            device = "auto", 
            n_envs: int = 1, 
            seed: int = None
        ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, seed)
        self.batch_size = batch_size

        if isinstance(self.obs_shape, dict):
            self.observations = {key: np.zeros([self.buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.obs_shape.items()}
        else:
            self.observations = np.zeros([buffer_size, n_envs, *self.obs_shape], dtype=np.float32)
        
        if isinstance(self.action_shape, dict):
            self.actions = {key: np.zeros([self.buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.action_shape.items()}
        else:
            self.actions = np.zeros([buffer_size, n_envs, *self.action_shape], dtype=np.float32)
        
        self.rewards = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.terminals = np.zeros([buffer_size, n_envs], dtype=np.bool_)

        self.log_probs = np.zeros([buffer_size, n_envs], dtype=np.float32)
        self.values = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.advantages = np.zeros([buffer_size, n_envs], dtype=np.float32)
        self.returns = np.zeros([buffer_size, n_envs], dtype=np.float32)


    def add(self, observation: np.ndarray | dict[str, np.ndarray], reward: np.ndarray, action: np.ndarray | dict[str, np.ndarray], value: np.ndarray, log_prob: np.ndarray, terminal: np.ndarray) -> None:
        
        idx = self.mem_cntr

        if isinstance(self.observations, dict):
            for key in observation.keys():
                self.observations[key][idx] = observation[key]
        else:
            self.observations[idx] = observation
        
        if isinstance(self.actions, dict):
            for key in action.keys():
                self.actions[key][idx] = action[key]
        else:
            self.actions[idx] = action
        
        self.values[idx] = value
        self.log_probs[idx] = log_prob

        self.rewards[idx] = reward
        self.terminals[idx] = terminal

        self.mem_cntr += 1

        if self.mem_cntr == self.buffer_size:
            self.full = True
            self.mem_cntr = 0
    
    def sample(self):
        if self.batch_size:
            ...
        else:
            ...
            
        return super().sample()