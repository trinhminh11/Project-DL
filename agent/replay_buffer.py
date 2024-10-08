from typing import Literal
import torch
from torch import Tensor
import numpy as np
from numpy._typing import _ShapeLike
import random


class ReplayBuffer:
    def __init__(self, state_shape: _ShapeLike, action_shape: _ShapeLike, batch_size: int | Literal[False], on_policy: bool, buffer_size: int = int(1e5), device: str = 'cpu', seed = 0):
        if batch_size is False and on_policy is False:
            raise ValueError("batch_size must be provided for off-policy agents.")
        
        self.mem_size = buffer_size

        self.state_memory = np.zeros([buffer_size, *state_shape], dtype=np.float32)
        self.action_memory = np.zeros([buffer_size, *action_shape], dtype=np.float32)
        self.reward_memory = np.zeros([buffer_size, 1], dtype=np.float32)
        self.next_state_memory = np.zeros([buffer_size, *state_shape], dtype=np.float32)
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

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = terminal

        self.mem_cntr += 1
        
        if self.length < self.mem_size:
            self.length = self.mem_cntr

        if self.mem_cntr == self.mem_size:
            self.mem_cntr = 0

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.on_policy:
            if self.batch_size is False:
                """Return the all experience from memory."""
                batch = range(self.mem_cntr - len(self), self.mem_cntr)

                states = torch.from_numpy(self.state_memory[batch]).to(self.device)
                actions = torch.from_numpy(self.action_memory[batch]).to(self.device)
                rewards = torch.from_numpy(self.reward_memory[batch]).to(self.device)
                next_states = torch.from_numpy(self.next_state_memory[batch]).to(self.device)
                terminals = torch.from_numpy(self.terminal_memory[batch]).bool().to(self.device)

            else:
                """Return the last batch of experiences from memory."""
                batch = range(self.mem_cntr - self.batch_size, self.mem_cntr)

                states = torch.from_numpy(self.state_memory[batch]).to(self.device)
                actions = torch.from_numpy(self.action_memory[batch]).to(self.device)
                rewards = torch.from_numpy(self.reward_memory[batch]).to(self.device)
                next_states = torch.from_numpy(self.next_state_memory[batch]).to(self.device)
                terminals = torch.from_numpy(self.terminal_memory[batch]).bool().to(self.device)

        else:
            """Randomly sample a batch of experiences from memory."""
            batch = random.sample(range(len(self)), self.batch_size)

            states = torch.from_numpy(self.state_memory[batch]).to(self.device)
            actions = torch.from_numpy(self.action_memory[batch]).to(self.device)
            rewards = torch.from_numpy(self.reward_memory[batch]).to(self.device)
            next_states = torch.from_numpy(self.next_state_memory[batch]).to(self.device)
            terminals = torch.from_numpy(self.terminal_memory[batch]).bool().to(self.device)

        return states, actions, rewards, next_states, terminals
    
    def clear(self):
        self.mem_cntr = 0
        self.length = 0

    def __len__(self):
        """Return the current size of internal memory."""
        return self.length
