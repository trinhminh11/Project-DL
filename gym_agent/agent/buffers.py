from typing import Literal, NamedTuple
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import Tensor
import numpy as np
from numpy._typing import _ShapeLike
import random
import gymnasium.spaces as spaces

from .. import utils

from numpy.typing import NDArray

# import psutil

# MAX_MEM_AVAILABLE = psutil.virtual_memory().available

class ReplayBufferSamples(NamedTuple):
    observations: Tensor | dict[str, Tensor]
    actions: Tensor | dict[str, Tensor]
    rewards: Tensor
    next_observations: Tensor
    terminals: Tensor

class RolloutBufferSamples(NamedTuple):
    observations: Tensor | dict[str, Tensor]
    actions: Tensor | dict[str, Tensor]
    rewards: Tensor
    values: Tensor
    log_prob: Tensor
    advantages: Tensor
    returns: Tensor

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
        
        self.buffer_size = buffer_size
        self.obs_shape = utils.get_shape(observation_space)
        self.action_shape = utils.get_shape(action_space)
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
    
    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.mem_cntr = 0
        self.full = False

class ReplayBuffer(BaseBuffer):

    observations: NDArray | dict[str, NDArray]
    actions: NDArray | dict[str, NDArray]
    rewards: NDArray
    terminals: NDArray

    def __init__(
            self, 
            buffer_size: int, 
            observation_space: spaces.Space, 
            action_space: spaces.Space, 
            device = "auto", 
            n_envs: int = 1, 
            seed: int = None
        ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs, seed)

        if isinstance(self.obs_shape, dict):
            self.observations = {key: np.zeros([buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.obs_shape.items()}
        else:
            self.observations = np.zeros([buffer_size, n_envs, *self.obs_shape], dtype=np.float32)
        
        if isinstance(self.action_shape, dict):
            self.actions = {key: np.zeros([buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.action_shape.items()}
        else:
            self.actions = np.zeros([buffer_size, n_envs, *self.action_shape], dtype=np.float32)
        
        self.rewards = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.terminals = np.zeros([buffer_size, n_envs], dtype=np.bool_)


    def add(
            self, 
            observation: NDArray | dict[str, NDArray], 
            action: NDArray | dict[str, NDArray], 
            reward: NDArray, 
            next_observation: NDArray | dict[str, NDArray], 
            terminal: NDArray
        ) -> None:
        """Add a new experience to memory."""
        idx = self.mem_cntr

        if isinstance(self.obs_shape, dict):
            for key in observation.keys():
                self.observations[key][idx] = observation[key]
                self.observations[key][(idx+1)%self.buffer_size] = next_observation[key]
        else:
            self.observations[idx] = observation
            self.observations[(idx+1)%self.buffer_size] = next_observation
        
        if isinstance(self.action_shape, dict):
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

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample a batch of experiences from memory."""

        if self.full:
            batch = (np.random.randint(1, self.buffer_size, size=batch_size) + self.mem_cntr) % self.buffer_size
        else:
            batch = np.random.randint(0, self.mem_cntr, size=batch_size)

        return self._get_sample(batch)
    
    def _get_sample(self, batch: NDArray[np.uint32]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        env_ind = np.random.randint(0, self.n_envs, size = len(batch))

        if isinstance(self.obs_shape, dict):
            observations = {key: obs[batch, env_ind, :] for key, obs in self.observations.items()}
            next_observations = {key: obs[(batch+1)%self.buffer_size, env_ind, :] for key, obs in self.observations.items()}
        else:
            observations = self.observations[batch, env_ind, :]
            next_observations = self.observations[(batch+1)%self.buffer_size, env_ind, :]
        
        if isinstance(self.action_shape, dict):
            actions = {key: act[batch, env_ind, :] for key, act in self.actions.items()}
        else:
            actions = self.actions[batch, env_ind, :]
        
        rewards = self.rewards[batch, env_ind]
        terminals = self.terminals[batch, env_ind]

        return ReplayBufferSamples(
            observations=utils.to_torch(observations, device=self.device),
            actions=utils.to_torch(actions, device=self.device),
            rewards=utils.to_torch(rewards, device=self.device),
            next_observations=utils.to_torch(next_observations, device=self.device),
            terminals=utils.to_torch(terminals, device=self.device)
        )

class RolloutBuffer(BaseBuffer):
    observations: NDArray | dict[str, NDArray]
    actions: NDArray | dict[str, NDArray]
    rewards: NDArray

    advantages: NDArray
    returns: NDArray
    log_probs: NDArray
    values: NDArray

    def __init__(
            self, 
            buffer_size: int, 
            observation_space: spaces.Space, 
            action_space: spaces.Space, 
            gamma = 0.99,
            gae_lambda = 0.95,
            device = "auto", 
            n_envs: int = 1, 
            seed: int = None
        ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, seed)

        if isinstance(self.obs_shape, dict):
            self.observations = {key: np.zeros([buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.obs_shape.items()}
        else:
            self.observations = np.zeros([buffer_size, n_envs, *self.obs_shape], dtype=np.float32)
        
        if isinstance(self.action_shape, dict):
            self.actions = {key: np.zeros([buffer_size, n_envs, *shape], dtype=np.float32) for key, shape in self.action_shape.items()}
        else:
            self.actions = np.zeros([buffer_size, n_envs, *self.action_shape], dtype=np.float32)
        
        self.rewards = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.log_probs = np.zeros([buffer_size, n_envs], dtype=np.float32)
        self.values = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.advantages = np.zeros([buffer_size, n_envs], dtype=np.float32)
        self.returns = np.zeros([buffer_size, n_envs], dtype=np.float32)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.dies = np.zeros((n_envs, ), dtype=np.bool_)
        self.end_mem_pos = np.zeros((n_envs, ), dtype=np.uint32)

        self.processed = False

    def add(self, observation: NDArray | dict[str, NDArray], action: NDArray | dict[str, NDArray], reward: NDArray, value: NDArray, log_prob: NDArray, terminal: NDArray[np.bool_]) -> None:
        if self.processed:
            raise ValueError("Cannot add new experiences to the buffer after processing the buffer.")
        
        idx = self.mem_cntr

        if isinstance(self.obs_shape, dict):
            for key in observation.keys():
                self.observations[key][idx, ~self.dies] = observation[key]
        else:
            self.observations[idx, ~self.dies] = observation
        
        if isinstance(self.action_shape, dict):
            for key in action.keys():
                self.actions[key][idx, ~self.dies] = action[key]
        else:
            self.actions[idx, ~self.dies] = action

        self.rewards[idx, ~self.dies] = reward
        
        self.values[idx, ~self.dies] = value
        self.log_probs[idx, ~self.dies] = log_prob

        self.mem_cntr += 1
        self.end_mem_pos += (~self.dies)

        
        self.dies = self.dies | terminal


        if self.mem_cntr == self.buffer_size:
            raise ValueError("Rollout buffer is full. Please reset the buffer.")
    
    def reset(self) -> None:
        super().reset()
        self.dies = np.zeros((self.n_envs, ), dtype=np.bool_)
        self.end_mem_pos = np.zeros((self.n_envs, ), dtype=np.uint32)
        self.processed = False

        self.observations = np.zeros([self.buffer_size, self.n_envs, *self.obs_shape], dtype=np.float32)
        self.actions = np.zeros([self.buffer_size, self.n_envs, *self.action_shape], dtype=np.float32)
        self.rewards = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.values = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.log_probs = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.advantages = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.returns = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)

    
    def calc_advantages_and_returns(self, last_values: NDArray[np.float32], last_terminals: NDArray[np.bool_]) -> None:
        if self.processed:
            raise ValueError("Cannot calculate advantages and returns after processing the buffer.")
                
        def _calc_advantages_and_returns(env):
            if self.gae_lambda == 1:
                # No GAE, returns are just discounted rewards
                T = self.end_mem_pos[env]
                advantages = np.zeros(T, dtype=np.float32)
                returns = np.zeros(T, dtype=np.float32)
                next_return = 0
                for t in reversed(range(T)):
                    next_return = self.rewards[t][env] + self.gamma * next_return
                    returns[t] = next_return

                advantages = returns - self.values[:T, env]


                return advantages, returns
                
            T = self.end_mem_pos[env]
            advantages = np.zeros(T, dtype=np.float32)
            last_gae_lambda = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - last_terminals[env]
                    next_values = last_values[env]
                else:
                    next_non_terminal = 1.0
                    next_values = self.values[t + 1][env]

                delta = self.rewards[t][env] + next_non_terminal * self.gamma * next_values - self.values[t][env]
                last_gae_lambda = delta + next_non_terminal*self.gamma * self.gae_lambda *  last_gae_lambda

                advantages[t] = last_gae_lambda 
            
            returns = advantages + self.values[:T, env]

            return advantages, returns


        for env in range(self.n_envs):
            advantages , returns = _calc_advantages_and_returns(env)
            self.advantages[:self.end_mem_pos[env], env] = advantages
            self.returns[:self.end_mem_pos[env], env] = returns
    
    def process_mem(self) -> NDArray:
        if self.processed:
            raise ValueError("Cannot process the buffer again.")
        
        def swap_and_flatten(arr: NDArray) -> NDArray:
            """
            Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
            to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
            to [n_envs * n_steps, ...] (which maintain the order)
            """
            ret_arr = np.zeros((sum(self.end_mem_pos), *arr.shape[2:]), dtype=arr.dtype)

            ret_arr[:self.end_mem_pos[0]] = arr[:self.end_mem_pos[0], 0]
            for i in range(1, self.n_envs):
                ret_arr[self.end_mem_pos[i-1]:self.end_mem_pos[i]] = arr[:self.end_mem_pos[i], i]

            return ret_arr

        
        self.observations = swap_and_flatten(self.observations)
        self.actions = swap_and_flatten(self.actions)
        self.rewards = swap_and_flatten(self.rewards)
        self.values = swap_and_flatten(self.values)
        self.log_probs = swap_and_flatten(self.log_probs)
        self.advantages = swap_and_flatten(self.advantages)
        self.returns = swap_and_flatten(self.returns)

        self.processed = True


    def get(self, batch_size: int = None):
        if not self.processed:
            self.process_mem()
        
        mem_size = sum(self.end_mem_pos)

        if batch_size is None or batch_size is False:
            yield self._get_sample(np.arange(mem_size))
        
        else:
            batch_size = mem_size

            indices = np.random.permutation(mem_size)

            for start_idx in range(0, mem_size, batch_size):
                yield self._get_sample(indices[start_idx : start_idx + batch_size])
    
    def sample(self, batch_size: int) -> RolloutBufferSamples:
        if not self.processed:
            self.process_mem()

        batch = np.random.randint(0, sum(self.end_mem_pos), size=batch_size)

        return self._get_sample(batch)
            
    def _get_sample(self, batch: NDArray[np.uint32]) -> RolloutBufferSamples:
        if isinstance(self.obs_shape, dict):
            observations = {key: self.observations[batch] for key in self.obs_shape.keys()}
        else:
            observations = self.observations[batch]
        
        if isinstance(self.action_shape, dict):
            actions = {key: self.actions[batch] for key in self.action_shape.keys()}
        else:
            actions = self.actions[batch]
        
        return RolloutBufferSamples(
            observations=utils.to_torch(observations, device=self.device),
            actions=utils.to_torch(actions, device=self.device),
            rewards = utils.to_torch(self.rewards[batch], device=self.device),
            values=utils.to_torch(self.values[batch], device=self.device),
            log_prob=utils.to_torch(self.log_probs[batch], device=self.device),
            advantages=utils.to_torch(self.advantages[batch], device=self.device),
            returns=utils.to_torch(self.returns[batch], device=self.device)
        )
    