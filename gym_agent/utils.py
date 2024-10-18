# Third-party imports
import torch
import torch.nn as nn

import gymnasium.spaces as spaces

import matplotlib.pyplot as plt


def get_device(device: str | torch.device = "auto") -> str:
    if not isinstance(device, str) and not isinstance(device, torch.device):
        raise ValueError(f"Invalid device: {device}")
    
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_shape(
    space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param space:
    :return:
    """
    if isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(space, spaces.MultiDiscrete):
        # Number of discrete features
        return (space.shape, 1)
    elif isinstance(space, spaces.MultiBinary):
        # Number of binary features
        return space.shape
    elif isinstance(space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in space.spaces.items()}  # type: ignore[misc]
    else:
        raise NotImplementedError(f"{space} space is not supported")

def check_for_nested_spaces(space: spaces.Space) -> None:
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param space: an observation space
    """
    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = space.spaces.values() if isinstance(space, spaces.Dict) else space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )




def to_device(*args, device='cuda'):
    for arg in args:
        arg.to(device)
