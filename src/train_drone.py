#!/usr/bin/env python3
"""
Training Script for Autonomous Drone RL Model

This script focuses on training the reinforcement learning model using
Stable-Baselines3 and PX4 SITL with Gazebo.

Usage:
    python train_drone.py --timesteps 100000 --connection udp://:14540
"""
import json
import os
import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from drone_env import DroneEnv
from gazebo_manager import GazeboManager


def load_px4_path_from_config():
    """
    Load PX4 directory from config with correct path resolution
    """
    config_path = os.path.join(os.path.dirname(__file__), "../_config/directory_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_content = f.read()
                config = json.loads(config_content)
                raw_path = config.get("px4-autopilot")
                if raw_path:
                    abs_path = os.path.abspath(os.path.expanduser(raw_path))
                    return abs_path
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print(f"Config file not found at {config_path}")

    default_path = os.path.expanduser("~/PX4-Autopilot")
    return default_path


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback for saving model when reward improves"""

    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            # Get the proper key if available
            if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
                # Get the data
                x, y = self.model.logger.name_to_value['rollout/ep_rew_mean']
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = y[-1]
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                    # New best model, save it
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)
            else:
                if self.verbose > 0:
                    print("Warning: rollout/ep_rew_mean not found in logger")

        return True


def create_training_env(connection_string="udp://:14540", max_retries=3):
    """Create and configure the training environment with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to create environment...")

            # Create the environment
            env = DroneEnv(connection_string=connection_string)
            # Test the environment with a reset to ensure the connection works
            env.reset()

            log_dir = "../logs"
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)

            # Use vectorized environment (even for single env)
            env = DummyVecEnv([lambda: env])

            # Normalize observations and rewards
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

            print("Environment created successfully!")
            return env

        except Exception as e:
            print(f"Error creating environment (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Waiting 5 seconds before next attempt...")
                time.sleep(5)
            else:
                print("Maximum retry attempts reached. Could not create environment.")
                raise


def train_model(env, timesteps=100000, save_dir="checkpoints", device="cpu"):
    """Train the RL model with specified device"""
    os.makedirs(save_dir, exist_ok=True)

    # Create tensorboard log directory
    tensorboard_log = "./tensorboard/"
    os.makedirs(tensorboard_log, exist_ok=True)

    print(f"Training model on device: {device}")

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        device=device,
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="drone_model"
    )

    reward_callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=save_dir
    )

    # Start training
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, reward_callback],
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Save normalized environment settings
    env_path = os.path.join(save_dir, "vec_normalize.pkl")
    env.save(env_path)
    print(f"Environment normalization saved to {env_path}")

    return model, final_model_path


def evaluate_model(model, env, n_eval_episodes=10):
    """Evaluate the trained model"""
    print(f"Evaluating model over {n_eval_episodes} episodes...")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def main():
    px4_default_dir = load_px4_path_from_config() or os.path.expanduser("~/PX4-Autopilot")

    parser = argparse.ArgumentParser(description="Train an autonomous drone RL model")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--connection", type=str, default="udp://:14540", help="MAVSDK connection string")
    parser.add_argument("--px4-dir", type=str, default=px4_default_dir, help="PX4 directory (default from config or ~/PX4-Autopilot)")
    parser.add_argument("--no-gazebo", action="store_true", help="Don't start Gazebo (assumes it's already running)")
    parser.add_argument("--eval", action="store_true", help="Evaluate model after training")
    args = parser.parse_args()

    # Start Gazebo if needed
    gazebo_manager = None
    if not args.no_gazebo:
        gazebo_manager = GazeboManager(px4_dir=args.px4_dir)
        gazebo_manager.start()

    try:
        # Wait for Gazebo and PX4 to initialize
        print("Waiting for simulation to stabilize...")
        time.sleep(5)

        # Create environment
        print("Creating training environment...")
        env = create_training_env(args.connection)

        # Train model
        model, model_path = train_model(env, args.timesteps)

        # Evaluate if requested
        if args.eval:
            evaluate_model(model, env)

        print(f"Training complete! Model saved to {model_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Clean up
        if 'env' in locals():
            env.close()

        if gazebo_manager:
            gazebo_manager.stop()


if __name__ == "__main__":
    main()
