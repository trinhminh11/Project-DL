from abc import ABC, abstractmethod
from typing import Any, Type
import gymnasium as gym
import numpy as np
from torch import Tensor
from tqdm import tqdm

from numpy._typing import _ShapeLike

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, max_size: int, batch_size: int, seed: int) -> None:
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_shape (int): dimension of each state
            action_shape (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        ...
    
    def to(self, device):
        """
        Moves the memory to the specified device.

        Args:
            device (str or torch.device): The device to move the memory to.
        """

    @abstractmethod
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

    @abstractmethod
    def sample(self) -> tuple:
        """
        Samples a batch of experiences from the replay buffer.

        Returns
        =======
            tuple: Tuple containing batches of states, actions, rewards, next_states, and terminals.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the current size of the replay buffer.

        Returns
        =======
            int: Current size of the replay buffer.
        """
        ...


class AgentBase(ABC):
    """
    Abstract base class for reinforcement learning agents.
    """

    def __init__(
        self,
        state_size: _ShapeLike,
        action_size: _ShapeLike,
        batch_size: int,
        max_mem_size: int,
        update_every: int,
        device: str,
        seed,
        **kwargs
    ) -> None:
        """
        Initializes the agent with the given parameters.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            batch_size (int): Size of each training batch
            max_mem_size (int): Maximum size of the replay buffer
            update_every (int): How often to update the network
            device (str): Device to use for tensor computations ('cpu' or 'cuda')
            seed (int): Random seed
            kwargs: Additional arguments.
        """
        ...

    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, terminal: np.ndarray) -> None:
        """
        Takes a step in the environment, storing the experience in the replay buffer and calling method:``learn`` every ``update_every`` step.

        Params
        ======
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            reward (np.ndarray): The reward received.
            next_state (np.ndarray): The next state.
            terminal (np.ndarray): Whether the episode has ended.
        """
        ...

    def reset(self) -> None:
        """
        Resets the agent at the start of an episode.
        This method should be called at the start of an episode (after `env.reset`).
        """
        ...

    @abstractmethod
    def act(self, state: np.ndarray) -> Any:
        """
        :param state: The current state.
        :return: The action to be taken.
        """
        ...

    @abstractmethod
    def learn(self, states: Tensor, actions: Tensor, rewards: Tensor, next_states: Tensor, terminals: Tensor) -> None:
        """
        Params
        ======
            states (Tensor): Batch of states.
            actions (Tensor): Batch of actions.
            rewards (Tensor): Batch of rewards.
            next_states (Tensor): Batch of next states.
            terminals (Tensor): Batch of terminal flags.
        """
        ...

    def fit(self, env: gym.Env, n_games: int, max_t: int, save_best=False, save_last=False, save_dir="./", progress_bar: Type[tqdm] = None) -> list:
        """
        Trains the agent over a number of games.

        Params
        ======
            env (gym.Env): Gym's Environment with `render_mode` = `None`.
            n_games (int): Number of games to simulate.
            max_t (int): Maximum steps per game.
            save_best (bool): Save the best agent.
            save_last (bool): Save the last agent (ignored if `save_best` is True).
            save_dir (str): Directory to save the agent.
            progress_bar (Type[tqdm]): Pass `tqdm` to enable progress bar.
        Returns
        =======
            list: List of scores.
        """
        ...

    def play(self, env: gym.Env) -> float:
        """
        Allows the agent to play a game and returns the total reward.

        :param env: Gym's Environment with `render_mode` = `human`.
        :return: Total reward obtained in the game.
        """
        ...
    
    def save(self, dir: str) -> None:
        """
        Saves the agent's state to a directory.

        :param dir: Directory where the agent's state will be saved.
        """
        ...

    def load(self, dir: str) -> None:
        """
        Loads the agent's state from a directory.

        :param dir: Directory from which the agent's state will be loaded.
        """
        ...