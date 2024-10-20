# Standard library imports
import random
import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Callable, Any, Generator
import typing

# Third-party imports
import gymnasium as gym
import numpy as np
from numpy._typing import _ShapeLike
from numpy.typing import NDArray
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from .buffers import ReplayBuffer, RolloutBuffer, BaseBuffer, RolloutBufferSamples

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
            device: str = 'auto',
            seed = None,
        ):
        utils.check_for_nested_spaces(env.observation_space)
        utils.check_for_nested_spaces(env.action_space)
        
        if not isinstance(policy, nn.Module):
            raise ValueError("policy must be an instance of torch.nn.Module")
        
        if isinstance(env, core.EnvWithTransform):
            if isinstance(env.unwrapped, gym.vector.VectorEnv):
                self.env = env
            else:
                self.env = core.EnvWithTransform(gym.vector.AsyncVectorEnv([lambda : env.env]))
        else:
            if isinstance(env, str):
                self.env = core.make_vec(env, 1)
            elif isinstance(env, gym.vector.VectorEnv):
                self.env = core.EnvWithTransform(env)
            elif isinstance(env, gym.Env):
                self.env = core.EnvWithTransform(gym.vector.AsyncVectorEnv([lambda : env]))
            else:
                raise ValueError("env must be an instance of EnvWithTransform, gym.Env, or str")


        self.n_envs = self.env.unwrapped.num_envs


        self.policy = policy

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
    def predict(self, observations: NDArray | dict[str, NDArray], deterministic: bool = True) -> ActType:
        """
        Perform an action based on the given observations.

        Parameters:
            observations (NDArray | dict[str, NDArray]): The input observations which can be either a numpy array or a dictionary
            * ``NDArray`` shape - `[batch, *obs_shape]`
            * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            deterministic (bool, optional): If True, the action is chosen deterministically. Defaults to True.

        Returns:
            ActType: The action to be performed.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    
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
            action = self.predict(np.expand_dims(obs, 0), True)

            next_obs, reward, terminated, truncated, info = env.step(action[0])

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
    
    def play_jupyter(self, env: core.EnvWithTransform, stop_if_truncated: bool = False, seed = None, FPS = 30) -> float:
        self.eval = True
        import pygame
        from IPython.display import display
        from PIL import Image

        pygame.init()
        clock = pygame.time.Clock()
        score = 0
        obs = env.reset(seed=seed)[0]
        self.reset()

        done = False
        while not done:
            clock.tick(FPS)
            env.render()
            action = self.predict(np.expand_dims(obs, 0), True)

            next_obs, reward, terminated, truncated, info = env.step(action[0])

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

        return score

class OffPolicyAgent(AgentBase):
    memory: ReplayBuffer
    def __init__(
            self, 
            policy: nn.Module, 
            env: core.EnvWithTransform | gym.Env | str, 
            buffer_size = int(1e5),
            batch_size: int = 64,
            update_every: int = 1,
            device = 'auto', 
            seed = None
        ):
        super().__init__(policy, env, device, seed)

        self.memory = ReplayBuffer(buffer_size = buffer_size, observation_space = self.env.observation_space, action_space = self.env.action_space, device = self.device, n_envs = self.n_envs, seed = self.seed)

        self.update_every = update_every

        self.batch_size = batch_size

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
    
    def step(self, observation: NDArray | dict, action: NDArray | dict, reward: NDArray, next_observation: NDArray | dict, terminal: NDArray):
        """
        Perform a single step in the agent's environment.
        This method adds the current experience to the replay buffer, increments the time step,
        and updates the agent by sampling a batch from memory if it's time to update.
        Args:
            observation (NDArray): The current state observation.
            action (NDArray): The action taken by the agent.
            reward (NDArray): The reward received after taking the action.
            next_observation (NDArray): The next state observation after taking the action.
            terminal (NDArray): A flag indicating whether the episode has ended.
        Returns:
            None
        """
        
        # Add the current experience to the replay buffer
        self.memory.add(observation, action, reward, next_observation, terminal)

        # Increment the time step and check if it's time to update
        self.time_step = (self.time_step + 1) % self.update_every

        # If it's time to update, sample a batch from memory and learn from it
        if self.time_step == 0:
            if len(self.memory) >= self.batch_size:
                observations, actions, rewards, next_observations, terminals = self.memory.sample(self.batch_size)
                self.learn(observations, actions, rewards, next_observations, terminals)


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
            
            action = self.predict(obs, deterministic)
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
                action = self.predict(obs, deterministic)
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


class OnPolicyAgent(AgentBase):
    memory: RolloutBuffer

    def __init__(
            self, 
            policy: nn.Module, 
            env: core.EnvWithTransform | gym.Env | str, 
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            buffer_size = int(1e5),
            batch_size: int = 64,
            device = 'auto', 
            seed = None
        ):
        super().__init__(policy, env, device, seed)

        self.memory = RolloutBuffer(buffer_size = buffer_size, observation_space = self.env.observation_space, action_space = self.env.action_space, device = self.device, n_envs = self.n_envs, gamma = gamma, gae_lambda = gae_lambda, seed = self.seed)

        self.batch_size = batch_size
    
    @abstractmethod
    def predict(self, state: NDArray | dict[str, NDArray], deterministic: bool = True) -> ActType:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state: NDArray | dict[str, NDArray], deterministic: bool = True) -> tuple[NDArray, NDArray, NDArray]:
        """
        Evaluate the given state and return the action, log probability, and value.

        Parameters:
            state (NDArray | dict[str, NDArray]): The input state which can be either a numpy array or a dictionary.
            deterministic (bool, optional): If True, the action is chosen deterministically. Defaults to True.

        Returns:
            tuple[NDArray, NDArray, NDArray]: A tuple containing:
                - actions (NDArray): The action to be performed.
                - values (NDArray): The estimated value of the state.
                - log_probs (NDArray): The log probability of the action.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, state: NDArray | dict[str, NDArray], action: NDArray | dict[str, NDArray]) -> tuple[NDArray, NDArray, NDArray]:
        """
        Evaluate the given actions for the provided states.

        Parameters:
            state (NDArray | dict[str, NDArray]): The input states which can be either a numpy array or a dictionary.
            action (NDArray | dict[str, NDArray]): The actions to be evaluated.

        Returns:
            tuple[NDArray, NDArray, NDArray]: A tuple containing:
                - values (NDArray): The estimated values of the states.
                - log_probs (NDArray): The log probabilities of the actions.
                - entropys (NDArray): The entropy of the action distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, rollout_generator: Generator[RolloutBuffer, None, None]) -> None:
        raise NotImplementedError
    
    def step(self, observation: NDArray | dict[str, NDArray], action: NDArray | dict[str, NDArray], reward: NDArray, value: NDArray, log_prob: NDArray, terminal: NDArray[np.bool_]):
        self.memory.add(observation, action, reward, value, log_prob, terminal)

    def fit(self, total_timesteps: int = None, n_games: int = None, deterministic=False, save_best=False, save_every=False, save_dir="./", progress_bar: Type[tqdm] = None, callbacks: Type[Callbacks] = None) -> list:
        if callbacks is None:
            callbacks = Callbacks(self)

        if total_timesteps is None and n_games is None:
            raise ValueError("Either total_timesteps or n_games must be provided")
        
        self.memory.reset()

        if total_timesteps:
            return self.fit_by_timesteps(total_timesteps, deterministic, save_best, save_every, save_dir, progress_bar, callbacks)
        else:
            return self.fit_by_games(n_games, deterministic, save_best, save_every, save_dir, progress_bar, callbacks)
    
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
                action, value, log_prob = self.evaluate(obs, deterministic)
                next_obs, reward, terminal, truncated, info = self.env.step(action)

                done = (terminal | truncated).all()
                self.step(obs, action, reward, value, log_prob, terminal)
                score += self.env.env_reward.mean()
                obs = next_obs
            
            self.memory.calc_advantages_and_returns(obs, terminal)

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
            
            callbacks.on_learn_begin()
            self.learn()
            callbacks.on_learn_end()
            self.memory.reset()

            if progress_bar:
                loop.set_postfix({"avg_score": avg_score, "score": scores[-1]})

        callbacks.on_train_end()

        return {'scores': scores, 'mean_scores': mean_scores, 'n_games': n_games, 'total_timesteps': total_timesteps}

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
            
            action, value, log_prob = self.act(obs, deterministic)
            next_obs, reward, terminal, truncated, info = self.env.step(action)
            done = (terminal or truncated).all()
            self.step(obs, action, reward, value, log_prob, terminal)
            score += self.env.env_reward.mean()
            obs = next_obs

            if done:
                self.memory.calc_advantages_and_returns(obs, terminal)
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
                
                callbacks.on_learn_begin()
                self.learn()
                callbacks.on_learn_end()
                self.memory.reset()

                if progress_bar:
                    loop.set_postfix({"game": episode, "avg_score": np.mean(scores[-100:]), "score": scores[-1]})

        callbacks.on_train_end()
        return {'scores': scores, 'mean_scores': mean_scores, 'n_games': episode, 'total_timesteps': total_timesteps}
