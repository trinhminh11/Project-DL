import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor, device

import utils

from agent import AgentBase

from numpy._typing import _ShapeLike

from typing import Type, Any


class Actor(nn.Module):
    def __init__(self, inp_channels: int, n_actions: int, features = [16, 32, 64, 128], name='actor', chkpt_dir = 'checkpoint/ddpg'):
        super().__init__()
        self.inp_channels = inp_channels
        self.n_actions = n_actions
        self.features = features
        self.name = name
        self.chkpt_dir = chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.checkpoint = os.path.join(self.chkpt_dir, self.name + ".pth")


        self.initial = nn.Sequential(
            nn.Conv2d(inp_channels, features[0], 3, 1, 1),
            nn.ReLU(True)
        )        

        self.net = nn.Sequential(
            *[utils.ConvBn(features[i], features[i+1], pool = (i!=(len(features)-2))) for i in range(len(features)-1)]
        )

        self.net.append(nn.AdaptiveMaxPool2d((1, 1)))
        self.net.append(nn.Flatten())

        self.mu = nn.Sequential(
            nn.Linear(features[-1], 64),
            nn.ReLU(True),
            nn.Linear(64, n_actions),
            nn.Tanh()
        )

    def forward(self, states: Tensor) -> Tensor:
        states_encoder = self.net(self.initial(states))

        mu = self.mu(states_encoder)

        return mu

    def save(self):
        torch.save(self.state_dict(), self.checkpoint)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint))

class Critic(nn.Module):
    def __init__(self, inp_channels: int, n_actions: int, features = [16, 32, 64, 128], name='critic', chkpt_dir = 'checkpoint/ddpg'):
        super().__init__()
        self.inp_channels = inp_channels
        self.n_actions = n_actions
        self.features = features
        self.name = name
        self.chkpt_dir = chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.checkpoint = os.path.join(self.chkpt_dir, self.name + ".pth")


        self.initial = nn.Sequential(
            nn.Conv2d(inp_channels, features[0], 3, 1, 1),
            nn.ReLU(True)
        )        

        self.net = nn.Sequential(
            *[utils.ConvBn(features[i], features[i+1], pool = (i!=(len(features)-2))) for i in range(len(features)-1)]
        )

        self.net.append(nn.AdaptiveMaxPool2d((1, 1)))
        self.net.append(nn.Flatten())

        self.q = nn.Sequential(
            nn.Linear(features[-1] + n_actions, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, states: Tensor, action: Tensor) -> Tensor:
        states_encoder = self.net(self.initial(states))

        action_value = self.q(torch.concat([states_encoder, action], 1))

        return action_value

    def save(self):
        torch.save(self.state_dict(), self.checkpoint)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint))

class DDPG(AgentBase):
    def __init__(
            self, 
            state_size: _ShapeLike, 
            action_size: _ShapeLike, 
            batch_size: int, 
            actor_lr = 1e-3,
            critic_lr = 2e-3,
            actor_optim: Type[optim.Optimizer] = optim.Adam,
            critic_optim: Type[optim.Optimizer] = optim.Adam,
            gamma = 0.99,
            tau = 5e-3,
            noise = 0.1,
            max_mem_size: int = int(1e5), 
            update_every: int = 1, 
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
            seed = 0, 
            **kwargs
        ) -> None:

        super().__init__(state_size, action_size, batch_size, max_mem_size, update_every, device, seed, **kwargs)

        self.gamma = gamma
        self.tau = tau
        self.noise = noise

        self.actor = Actor(state_size[0], action_size[0]).to(device)
        self.critic = Critic(state_size[0], action_size[0]).to(device)
        self.target_actor = Actor(state_size[0], action_size[0], name = 'target_actor').to(device)
        self.target_critic = Critic(state_size[0], action_size[0], name = 'target_critic').to(device)

        self.actor_optim = actor_optim(self.actor.parameters(), actor_lr)
        self.critic_optim = critic_optim(self.critic.parameters(), critic_lr)

        self.update_network_parameters(tau = 1)

    def to(self):
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)
        self.memory.to(self.device)
    
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self):
        print("-----saving-----")
        self.actor.save()
        self.critic.save()
        self.target_actor.save()
        self.target_critic.save()

    def load(self):
        print("-----loading-----")
        self.actor.load()
        self.critic.load()
        self.target_actor.load()
        self.target_critic.load()

    @torch.no_grad()
    def act(self, state: np.ndarray, eval = True) -> Any:
        if self.state_transform:
            state = self.state_transform(state)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.actor.eval()
        action: np.ndarray = self.actor(state)[0].cpu().detach().numpy()
        self.actor.train()

        if not eval:
            action += np.random.normal(0, self.noise, action.shape).clip(-1, 1)

        return np.array([action[0], action[1], 0]) if action[1]>0 else np.array([action[0], 0, -action[1]])

    def learn(self, states: Tensor, actions: Tensor, rewards: Tensor, next_states: Tensor, terminals: Tensor):
        target_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states, target_actions)
        q_value = self.critic(states, actions)

        target_q_value = rewards + self.gamma * next_q * (~terminals)

        critic_loss = F.mse_loss(q_value, target_q_value)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        new_policy_actions = self.actor(states)
        actor_loss = -self.critic(states, new_policy_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_network_parameters()


def main():
    import gymnasium as gym
    from tqdm import tqdm

    env = gym.make('CarRacing-v2')

    agent = DDPG(
        gamma=0.99,
        batch_size=64,
        state_size=(3, 96, 96),
        action_size=(2, ),
    )

    state_tfm = utils.Compose(
        utils.WHC2CWH(),
        type('Div255', (), {'__call__': lambda self, X: X/255, '__repr__': lambda self: 'Div255()'})(),
        utils.Normalize(0.5, 0.5)
    )

    agent.set_state_transform(state_tfm)
    agent.set_action_transform(utils.CarRacingAction())

    agent.fit(env, 500, 1000, tqdm)

if __name__ == "__main__":
    main()