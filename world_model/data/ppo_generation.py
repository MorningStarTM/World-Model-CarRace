import gym
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy



def data_generation(env_name, 
                    output_folder:str,
                    num_episodes:int = 100, 
                    steps_per_episode:int = 500,
                    repo_id:str = "Amankankriya/ppo-CarRacing-v2",
                    filename:str="ppo-CarRacing-v2.zip"
                    ):
    env = gym.make(env_name, render_mode='rgb_array')

    # Model loading
    checkpoint = load_from_hub(
    repo_id= repo_id,
    filename=filename,
    )
    model = PPO.load(checkpoint)

    # Generate data
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        obs = env.reset()
        for step in range(steps_per_episode):
            # Render environment and save the observation as an image
            img = env.render()  # Render returns a list of images if mode is 'rgb_array'

            # Check if img is a list and extract the first element if so
            if isinstance(img, list):
                img = img[0]

            # Save the observation as an image
            img = Image.fromarray(img)
            img.save(os.path.join(output_folder, f'images{episode}_{step}.png'))

            # Take a random action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            # Break the loop if the episode is done
            if done:
                break

    env.close()
    print("done")