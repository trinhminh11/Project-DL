from typing import Literal
import torch
from torch import Tensor
import numpy as np
from numpy._typing import _ShapeLike
import random


class ReplayBuffer:
    def __init__(self, state_shape: dict[str, _ShapeLike] | _ShapeLike, action_shape: _ShapeLike, batch_size: int | Literal[False], on_policy: bool, buffer_size: int = int(1e5), device: str = 'cpu', seed = 0):
        if batch_size is False and on_policy is False:
            raise ValueError("batch_size must be provided for off-policy agents.")
        
        self.mem_size = buffer_size

        if isinstance(state_shape, dict):
            self.state_memory = {key: np.zeros([buffer_size, *shape], dtype=np.float32) for key, shape in state_shape.items()}
            self.next_state_memory = {key: np.zeros([buffer_size, *shape], dtype=np.float32) for key, shape in state_shape.items()}
        else:
            self.state_memory = np.zeros([buffer_size, *state_shape], dtype=np.float32)
            self.next_state_memory = np.zeros([buffer_size, *state_shape], dtype=np.float32)

        self.action_memory = np.zeros([buffer_size, *action_shape], dtype=np.float32)
        self.reward_memory = np.zeros([buffer_size, 1], dtype=np.float32)
        self.terminal_memory = np.zeros([buffer_size, 1], dtype=np.bool_)

        self.batch_size = batch_size
        self.device = device
        self.on_policy = on_policy
        self.seed = random.seed(seed)

        self.mem_cntr = 0

        self.length = 0

    def to(self, device):
        self.device = device

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, terminal: np.ndarray):
        """Add a new experience to memory."""
        idx = self.mem_cntr

        if isinstance(self.state_memory, dict):
            for key in state.keys():
                self.state_memory[key][idx] = state[key]
                self.next_state_memory[key][idx] = next_state[key]

        else:
            self.state_memory[idx] = state
            self.next_state_memory[idx] = next_state

        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = terminal

        self.mem_cntr += 1
        
        if self.length < self.mem_size:
            self.length = self.mem_cntr

        if self.mem_cntr == self.mem_size:
            self.mem_cntr = 0

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        def _sample(batch):
            if isinstance(self.state_memory, dict):
                states = {key: torch.from_numpy(self.state_memory[key][batch]).to(self.device) for key in self.state_memory.keys()}
                next_states = {key: torch.from_numpy(self.next_state_memory[key][batch]).to(self.device) for key in self.next_state_memory.keys()}
            else:
                states = torch.from_numpy(self.state_memory[batch]).to(self.device)
                next_states = torch.from_numpy(self.next_state_memory[batch]).to(self.device)
    
            actions = torch.from_numpy(self.action_memory[batch]).to(self.device)
            rewards = torch.from_numpy(self.reward_memory[batch]).to(self.device)
            terminals = torch.from_numpy(self.terminal_memory[batch]).bool().to(self.device)

            return states, actions, rewards, next_states, terminals
        
        if self.on_policy:
            if self.batch_size is False:
                """Return the all experience from memory."""
                batch = range(self.mem_cntr - len(self), self.mem_cntr)
            else:
                """Return the last batch of experiences from memory."""
                batch = range(self.mem_cntr - self.batch_size, self.mem_cntr)
        else:
            """Randomly sample a batch of experiences from memory."""
            batch = random.sample(range(len(self)), self.batch_size)

        return _sample(batch)
    
    def clear(self):
        self.mem_cntr = 0
        self.length = 0

    def __len__(self):
        """Return the current size of internal memory."""
        return self.length
