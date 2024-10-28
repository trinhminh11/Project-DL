import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import gym_agent as ga
import gymnasium as gym
import numpy as np

from PIL import Image

class CarRacingRays(ga.Transform):
    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(14, ), 
            dtype=np.float32
        )

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        rays = observation['rays'].astype(np.float32)
        vels = observation['vels'].astype(np.float32)

        res = np.concatenate([rays, vels], axis=-1)
        return res

class CarRacingFrameStack(ga.Transform):
    def __init__(self, n_frame=4):
        super().__init__()

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=1, shape=(n_frame, 96, 96), dtype=np.float32),
            'vector': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7, ), dtype=np.float32),
        })

        self.n_frame = n_frame

        self.frames = {'image': np.zeros((self.n_frame, 96, 96), dtype=np.float32), 'vector': np.zeros((7, ), dtype=np.float32)}

        self.start = False

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        image = observation['image'].astype(np.float32).transpose([2, 0, 1]) # n_envs, 3, 96, 96
        r, g, b = image[0], image[1], image[1]
        gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b

        vels = observation['vels'].astype(np.float32)
        # res = np.concatenate([rays, vels], axis=-1)

        if self.start:
            self.frames['image'][0] = gray_image
            self.frames['image'] = np.roll(self.frames['image'], shift=-1, axis=0)
            
        else:
            for i in range(self.n_frame):
                self.frames['image'][i] = gray_image

        self.frames['vector'] = vels

        self.start = True

        return self.frames

    def reset(self, **kwargs):
        self.frames = {'image': np.zeros((self.n_frame, 96, 96), dtype=np.float32), 'vector': np.zeros((self.n_frame, 7), dtype=np.float32)}

        self.idx = 0

        self.start = False

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


class ImageAsObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(96, 96, 3), dtype=np.float32)

    def observation(self, observation):
        image = self.env.unwrapped.render()
        image = Image.fromarray(image)
        image = image.resize((96, 96))
        return np.array(image)


def set_random_seed(seed: int) -> None:
	"""
	Sets the seeds at a certain value.
	:param seed: the value to be set
	"""
	print("Setting seeds ...... \n")
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     

def init_weights(init_type='xavier'):
    def xavier(m: nn.Module):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)
    
    def kaiming(m: nn.Module):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
    
    if init_type == 'xavier':
        return xavier
    
    elif init_type == 'kaiming':
        return kaiming
    else:
        raise NotImplementedError(f'Initialization method {init_type} is not implemented')


def plotting(filename = None, **kwargs):
    for name, value in kwargs.items():
        plt.plot(value, label=name)
        plt.title(name)

    plt.legend()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
        
		
