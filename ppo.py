import gym_agent as ga

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

class PPO(ga.AgentBase):
    def __init__(
            self, 
            state_shape, 
            action_shape,
            n_action,
            policy: ga.BasePolicy,
            n_epochs,
            gamma = 0.99,
            lr = 3e-5,
            gae_lambda = 0.9,
            policy_clip = 0.2,
            batch_size = 64, 
            device = 'cuda', 
            **kwargs
        ):
        super().__init__(state_shape, action_shape, batch_size, True, device=device,**kwargs)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip

        self.lr = lr

        self.policy = policy

        self.optimizer = optim.Adam(self.policy.parameters(), lr)

    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        probs: torch.Tensor = self.policy.action(state)
        action = Categorical(probs).sample()
        return action.detach().cpu().numpy()[0]
    
    def learn(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, terminals: torch.Tensor):
        states = states.detach().to(self.device)
        actions = actions.detach().squeeze(1).to(self.device)
        rewards = rewards.detach().squeeze(1).to(self.device)
        
        old_log_probs = Categorical(self.policy.action(states)).log_prob(actions).detach().to(self.device)

        advantages, returns = self.calc_advantages_and_returns(states, rewards, terminals, next_states[-1])


        for _ in range(self.n_epochs):
            new_action_probs, critic_value = self.policy(states)

            new_log_probs: torch.Tensor = Categorical(new_action_probs).log_prob(actions)

            prob_ratio = (new_log_probs - old_log_probs).exp()
            weighted_probs = prob_ratio*advantages

            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                    1+self.policy_clip)*advantages
            
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            critic_loss = F.smooth_l1_loss(returns, critic_value.squeeze(1)).mean()

            total_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        
    def calc_advantages_and_returns(self, states: torch.Tensor, rewards: torch.Tensor, terminals: bool, last_state: torch.Tensor):
        T = len(states)
        
        values: torch.Tensor = self.policy.value(torch.cat([states, last_state.unsqueeze(0)])).squeeze(-1)

        advantages = torch.zeros((T, ), device=self.device)

        with torch.no_grad():
            lastgaelam = 0
            for t in reversed(range(T)):
                delta = rewards[t] - values[t] + self.gamma * (~terminals[t]) * values[t+1]

                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * (~terminals[t]) * lastgaelam

            advantages = (advantages - advantages.mean()) / advantages.std()

            returns = advantages + values[:-1]
            
            returns = (returns - returns.mean()) / returns.std()

        return advantages, returns
