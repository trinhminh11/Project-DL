from typing import Literal

import numpy as np
from numpy._typing import _ShapeLike

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, state_shape: _ShapeLike, action_shape: _ShapeLike, batch_size: int | Literal[False], on_policy: bool, buffer_size: int = int(1e5), device: str = 'cpu', seed = 0):
        """
        Initialize the agent.

        Params
        ======
            state_shape (_ShapeLike): The shape of the state space.
            action_shape (_ShapeLike): The shape of the action space.
            batch_size (int | Literal[False]): The size of the batch for training, or False if not applicable.
                - If the replay buffer is off-policy, batch_size must be provided (not False).
                - If the replay buffer is on-policy, batch_size = False means using all memory of that episode.
            on_policy (bool, optional): Whether the agent is on-policy.
            buffer_size (int): The size of the replay buffer. Defaults to 1e5.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Defaults to 'cpu'.
            seed (int, optional): The random seed for reproducibility. Defaults to 0.

        """
        ...
    
    def to(self, device):
        """
        Moves the memory to the specified device.

        Args:
            device (str or torch.device): The device to move the memory to.
        """

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, terminal: np.ndarray) -> None:
        """
        Adds a new experience to the replay buffer.

        Params
        ======
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            reward (np.ndarray): The reward received.
            next_state (np.ndarray): The next state.
            terminal (np.ndarray): Whether the episode has ended.
        """
        ...

    def sample(self) -> tuple:
        """
        Samples a batch of experiences from the replay buffer.

        Returns
        =======
            tuple: Tuple containing batches of states, actions, rewards, next_states, and terminals.
        """
        ...

    def __len__(self) -> int:
        """
        Returns the current size of the replay buffer.

        Returns
        =======
            int: Current size of the replay buffer.
        """
        ...