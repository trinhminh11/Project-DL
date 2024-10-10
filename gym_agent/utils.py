# Standard library imports
from typing import Any

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec


class ConvBn(nn.Module):
    r'''
    ``Convolution`` + ``BatchNorm`` + ``ReLu`` (+ ``MaxPool``)

    keeping the size of input, if ``Maxpool``, reduce the size by half
    '''
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, pool=False) -> None:
        r'''
        ``Convolution`` + ``BatchNorm`` + ``ReLu`` (+ ``MaxPool``)

        keeping the size of input, if ``Maxpool``, reduce the size by half
        '''
        super().__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
        if pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.Identity()

    def forward(self, X: Tensor):
        out = self.Conv(X)
        out = self.Bn(out)
        out = self.act(out)
        out = self.pool(out)
        return out

class Transform:
    """
    A base class for transformations.
    Methods
    -------
    reset(**kwargs)
        Resets the transformation parameters. This method should be overridden by subclasses.
    __call__(*args, **kwargs)
        Applies the transformation. This method must be implemented by subclasses.
    __repr__()
        Returns a string representation of the transformation class.
    """
    def reset(self, **kwargs): ...

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Transform must be implemented')
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(Transform):
    """
    Compose multiple Transform objects into a single transform.
    Args:
        *args: Variable length argument list of Transform objects.
    Attributes:
        args (list[Transform]): List of Transform objects.
    Methods:
        append(tfm: Transform):
            Append a Transform object to the list of transforms.
        __call__(X: np.ndarray):
            Apply all the transforms in sequence to the input array X.
        __repr__():
            Return a string representation of the Compose object.
    """
    def __init__(self, *args):
        """
        Initialize the object with a list of Transform instances.
        Args:
            *args: Variable length argument list, each should be an instance of Transform.
        Raises:
            TypeError: If any argument is not an instance of Transform.
        """
        self.args: list[Transform] = []
        for tfm in args:
            self.append(tfm)
    
    def append(self, tfm: Transform):
        """
        Appends a Transform object to the args list.
        Parameters:
            tfm (Transform): The Transform object to be appended.
        Raises:
            TypeError: If the provided tfm is not an instance of Transform.

        """
        if not isinstance(tfm, Transform):
            raise TypeError('tfm must be Transform')
        
        self.args.append(tfm)
    
    def __call__(self, X: np.ndarray):
        for tfm in self.args:
            X = tfm(X)
        
        return X

    def __repr__(self):
        res = 'Compose(\n'
        for tfm in self.args:
            res += f'\t{tfm}\n'
        res += ')'
        return res

class Normalize(Transform):
    """
    A class used to normalize a numpy array by subtracting the mean and dividing by the standard deviation.

    Attributes
    ----------
    mean : float
        The mean value to subtract from the array.
    std : float
        The standard deviation value to divide the array by.

    Methods
    -------
    __call__(X: np.ndarray) -> np.ndarray
        Applies normalization to the input numpy array.
    
    __repr__()
        Returns a string representation of the Normalize object.
    """
    def __init__(self, mean, std):
        """
        Initializes the instance with mean and standard deviation.

        Args:
            mean (float): The mean value.
            std (float): The standard deviation value.
        """
        self.mean = mean
        self.std = std

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std})'

class EnvWithTransform(gym.Wrapper):
    """
    A wrapper for gym environments that allows for transformations on observations, actions, and rewards.
    
    Attributes
    =====
        observation_transform (Transform): A transformation to apply to observations.
        action_transform (Transform): A transformation to apply to actions.
        reward_transform (Transform): A transformation to apply to rewards.
        
    Methods
    =====
        set_observation_transform(tfm: Transform):
            Sets the transformation to apply to observations.
        set_action_transform(tfm: Transform):
            Sets the transformation to apply to actions.
        set_reward_transform(tfm: Transform):
            Sets the transformation to apply to rewards.
        observation(observation):
            Applies the observation transformation if set.
        action(action):
            Applies the action transformation if set.
        reward(reward):
            Applies the reward transformation if set.
        step(action):
            Steps the environment with the transformed action and returns the transformed observation, reward, and other info.
        reset(**kwargs) -> tuple[Any, dict[str, Any]]:
            Resets the environment and the transformations if set.
    """
    def __init__(self, env: gym.Env):
        """
        Initializes the utility class with the given environment.

        Args:
            env: The environment to be used by the utility class.

        Attributes:
            observation_transform: A transformation function for observations, initialized to None.
            action_transform: A transformation function for actions, initialized to None.
            reward_transform: A transformation function for rewards, initialized to None.
        """
        super().__init__(env)
        self.observation_transform = None
        self.action_transform = None
        self.reward_transform = None

    def set_observation_transform(self, tfm: Transform):
        """
        Sets the observation transform for the environment.
        Parameters:
            tfm (Transform): An instance of the Transform class. This transform will be applied to the observations.
        Raises:
            TypeError: If the provided tfm is not an instance of the Transform class.
        Notes:
            If the provided transform has an 'observation_space' attribute, it will be assigned to the environment's observation_space.
        """
        if not isinstance(tfm, Transform):
            raise TypeError('observation_transform must be Transform')
        
        if hasattr(tfm, 'observation_space'):
            self.observation_space = tfm.observation_space

        self.observation_transform = tfm

    def set_action_transform(self, tfm: Transform):
        """
        Sets the action transform for the object.

        Parameters:
            tfm (Transform): The transform to be set. Must be an instance of the Transform class.

        Raises:
            TypeError: If the provided tfm is not an instance of the Transform class.

        Notes:
            If the provided transform has an 'action_space' attribute, it will be assigned to the object's 'action_space'.
        """
        if not isinstance(tfm, Transform):
            raise TypeError('action_transform must be Transform')

        if hasattr(tfm, 'action_space'):
            self.action_space = tfm.action_space        

        self.action_transform = tfm
    
    def set_reward_transform(self, tfm: Transform):
        """
        Sets the reward transformation function for the environment.
        Parameters:
            tfm (Transform): An instance of the Transform class that defines the reward transformation.
        Raises:
            TypeError: If the provided tfm is not an instance of the Transform class.
        Notes:
            If the provided Transform instance has a 'reward_range' attribute, it will be used to set the environment's reward range.
        """
        if not isinstance(tfm, Transform):
            raise TypeError('reward_transform must be Transform')
        
        if hasattr(tfm, 'reward_range'):
            self.reward_range = tfm.reward_range

        self.reward_transform = tfm

    def observation(self, observation):
        if self.observation_transform:
            return self.observation_transform(observation)
        return observation

    def action(self, action):
        if self.action_transform:
            return self.action_transform(action)
        return action
    
    def reward(self, reward):
        if self.reward_transform:
            return self.reward_transform(reward)
        return reward 

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        .. versionchanged:: 0.26

            The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
            to users when the environment had terminated or truncated which is critical for reinforcement learning
            bootstrapping algorithms.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        return self.observation(observation), self.reward(reward), terminated, truncated, info

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        if self.observation_transform:
            self.observation_transform.reset(**kwargs)
        
        if self.action_transform:
            self.action_transform.reset(**kwargs)
        
        if self.reward_transform:
            self.reward_transform.reset(**kwargs)

        obs, info = self.env.reset(**kwargs)

        return self.observation(obs), info
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def Discrete2Box(action_space: gym.spaces.Discrete) -> gym.spaces.Box:
    if isinstance(action_space, gym.spaces.Box):
        return action_space
    elif isinstance(action_space, gym.spaces.Discrete):
        return gym.spaces.Box(low=0, high=1, shape=(action_space.n, ))
    else:
        raise TypeError('action_space must be gym.spaces.Discrete or gym.spaces.Box')
    

def make(
        id: str | EnvSpec,
        max_episode_steps: int | None = None,
        autoreset: bool | None = None,
        apply_api_compatibility: bool | None = None,
        disable_env_checker: bool | None = None,
        **kwargs: Any,
    ):
    """
    Creates an gymnasium environment with the specified id and wraps it with EnvTransform.

    To find all available environments use ``gymnasium.envs.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id or a :class:`EnvSpec`. Optionally if using a string, a module to import can be included, e.g. ``'module:Env-v0'``.
            This is equivalent to importing the module first to register the environment followed by making the environment.
        max_episode_steps: Maximum length of an episode, can override the registered :class:`EnvSpec` ``max_episode_steps``.
            The value is used by :class:`gymnasium.wrappers.TimeLimit`.
        autoreset: Whether to automatically reset the environment after each episode (:class:`gymnasium.wrappers.AutoResetWrapper`).
        apply_api_compatibility: Whether to wrap the environment with the :class:`gymnasium.wrappers.StepAPICompatibility` wrapper that
            converts the environment step from a done bool to return termination and truncation bools.
            By default, the argument is None in which the :class:`EnvSpec` ``apply_api_compatibility`` is used, otherwise this variable is used in favor.
        disable_env_checker: If to add :class:`gymnasium.wrappers.PassiveEnvChecker`, ``None`` will default to the
            :class:`EnvSpec` ``disable_env_checker`` value otherwise use this value will be used.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment with wrappers applied.

    Raises:
        Error: If the ``id`` doesn't exist in the :attr:`registry`
    """
    return EnvWithTransform(gym.make(id, max_episode_steps, autoreset, apply_api_compatibility, disable_env_checker, **kwargs))

def to_device(*args, device='cuda'):
    for arg in args:
        arg.to(device)

def plotting(X, filename = None, **kwargs):
    for name, value in kwargs.items():
        plt.plot(X, value, label=name)
        plt.title(name)

    plt.legend()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()