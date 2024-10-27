from better_car_racing.car_racing import BetterCarRacing
from gymnasium.envs.box2d.car_racing import CarRacing
import numpy as np
import pygame
import gymnasium as gym

import better_car_racing

import gym_agent as ga

def main():
    n_agents = 1
    pygame.init()
    a = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_agents, 1))
    # print(a.shape)
    # a = np.array([0.0, 0.0, 0.0])

    def register_input():
        nonlocal quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[1][0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[1][0] = +1.0
                if event.key == pygame.K_UP:
                    a[1][1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[1][2] = +0.8  # set 1.0 for wheels to block to zero rotation
                
                if event.key == pygame.K_a:
                    a[0][0] = -1.0
                if event.key == pygame.K_d:
                    a[0][0] = +1.0
                if event.key == pygame.K_w:
                    a[0][1] = +1.0
                if event.key == pygame.K_s:
                    a[0][2] = +0.8  # set 1.0 for wheels to block to zero rotation
                
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True
                    

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[1][0] = 0
                if event.key == pygame.K_RIGHT:
                    a[1][0] = 0
                if event.key == pygame.K_UP:
                    a[1][1] = 0
                if event.key == pygame.K_DOWN:
                    a[1][2] = 0
                
                if event.key == pygame.K_a:
                    a[0][0] = 0
                if event.key == pygame.K_d:
                    a[0][0] = 0
                if event.key == pygame.K_w:
                    a[0][1] = 0
                if event.key == pygame.K_s:
                    a[0][2] = 0  # set 1.0 for wheels to block to zero rotation

            if event.type == pygame.QUIT:
                quit = True

    env = BetterCarRacing(render_mode="human", num_agents=n_agents, render_ray=True)

    print(env.action_space)
    print(a[0])

    # env = ga.make('BetterCarRacing-v0', render_mode="human", num_agents=n_agents, render_ray=True)

    # env = ga.make_vec('BetterCarRacing-v0', render_mode="human", num_agents=2, render_ray=True)
    min_vels = [float('inf') for _ in range(7)]
    max_vels = [float('-inf') for _ in range(7)]

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            if n_agents == 1:
                s, r, terminated, truncated, info = env.step(a[0])
            else:
                s, r, terminated, truncated, info = env.step(a)
            
            env.render()

            total_reward += r
            # if steps % 200 == 0 or terminated or truncated:
                # print("\naction " + str([f"{x:+0.2f}" for x in a]))
                # print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                print(total_reward)
                break
    env.close()

def test():
    import matplotlib.pyplot as plt
    env = BetterCarRacing(2, render_mode="rgb_array", random_direction=False)

    # env = ga.make_vec('BetterCarRacing-v0', 2, render_mode="rgb_array", num_agents=1)
    # print(env.track)
    # env = ga.make('BetterCarRacing-v0', render_mode="rgb_array", num_agents=1, random_di)
    obs = env.reset()[0]

    # print(obs['image'].shape)
    # print(env.observation_space['image'].shape)

    obs, rewards, terminated, truncated, _ = env.step(env.action_space.sample())

    print(obs['image'].shape, obs['rays'].shape, obs['vels'].shape)
    print(obs['direction'])

    # print(rewards.shape)
    # print(terminated)
    # print(truncated)
    # # plt.subplot(1, 2, 1)

    # plt.imshow(obs['image'][0][0])

    # plt.subplot(1, 2, 2)

    # plt.imshow(obs['image'][0][1])

    # plt.show()
    # print(env.track[0])

if __name__ == "__main__":
    main()
    # test()