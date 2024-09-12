import gym
import os
import numpy as np
from PIL import Image

# Create the environment
env = gym.make('CarRacing-v2', render_mode='rgb_array')

# Specify the folder to save images
output_folder = 'car_racing_images'
os.makedirs(output_folder, exist_ok=True)

# Define the number of episodes and steps per episode
num_episodes = 10
steps_per_episode = 100

# Generate data
for episode in range(num_episodes):
    obs = env.reset()
    for step in range(steps_per_episode):
        # Render environment and save the observation as an image
        img = env.render()
        
        # Save the observation as an image
        img = Image.fromarray(img)
        img.save(os.path.join(output_folder, f'episode_{episode}_step_{step}.png'))

        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Break the loop if the episode is done
        if done or truncated:
            break

env.close()
