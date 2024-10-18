from typing import Any, Sequence, Callable

from .policy import *
from .transforms import *

from gymnasium.envs.registration import EnvSpec

from gymnasium import Env, Wrapper

import gymnasium as gym

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

def make_vec(
        id: str | EnvSpec, 
        num_envs: int = 1, 
        max_episode_steps: int | None = None,
        autoreset: bool | None = None,
        apply_api_compatibility: bool | None = None,
        disable_env_checker: bool | None = None,
        vector_kwargs: dict[str, Any] | None = None,
        **kwargs: Any):
    
    if num_envs <= 0:
        raise ValueError("num_envs must be greater than 0")

    if vector_kwargs is None:
        vector_kwargs = {}

    envs = [lambda: gym.make(id, max_episode_steps, autoreset, apply_api_compatibility, disable_env_checker, **kwargs) for _ in range(num_envs)]
    
    return EnvWithTransform(gym.vector.AsyncVectorEnv(envs, **vector_kwargs))
    
