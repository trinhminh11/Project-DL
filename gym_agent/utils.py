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

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def to_device(*args, device='cuda'):
    for arg in args:
        arg.to(device)
