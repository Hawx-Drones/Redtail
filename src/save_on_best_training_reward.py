import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
            # Check all possible keys that might contain episode rewards
            possible_keys = [
                'rollout/ep_rew_mean',  # Most common key
                'ep_rew_mean',  # Alternative format
                'episode_reward'  # Another possible format
            ]

            # Find the first key that exists in logger
            reward_key = None
            for key in possible_keys:
                if key in self.model.logger.name_to_value:
                    reward_key = key
                    break

            if reward_key is not None:
                # Get the data for the available key
                try:
                    x, y = self.model.logger.name_to_value[reward_key]
                    if len(x) > 0:
                        # Mean training reward over the last 100 episodes
                        mean_reward = y[-1]
                        if self.verbose > 0:
                            print(f"Num timesteps: {self.num_timesteps}")
                            print(
                                f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                        # New best model, save it
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            if self.verbose > 0:
                                print(f"Saving new best model to {self.save_path}")
                            self.model.save(self.save_path)
                except Exception as e:
                    print(f"Error accessing reward data: {e}")
            else:
                # If no standard keys found, try to directly access statistics through model
                try:
                    # Try to access last episode rewards directly from rollout buffer
                    if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                        if hasattr(self.model.rollout_buffer, 'rewards'):
                            # Calculate mean reward from recent episodes
                            recent_rewards = self.model.rollout_buffer.rewards
                            if len(recent_rewards) > 0:
                                mean_reward = float(np.mean(recent_rewards))
                                if self.verbose > 0:
                                    print(f"Num timesteps: {self.num_timesteps}")
                                    print(
                                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward (from buffer): {mean_reward:.2f}")

                                # New best model, save it
                                if mean_reward > self.best_mean_reward:
                                    self.best_mean_reward = mean_reward
                                    if self.verbose > 0:
                                        print(f"Saving new best model to {self.save_path}")
                                    self.model.save(self.save_path)
                            else:
                                print("No rewards found in rollout buffer")
                        else:
                            # Use any other available statistics
                            if self.verbose > 0:
                                print("Warning: Cannot find rewards in rollout buffer")
                    else:
                        if self.verbose > 0:
                            print("Warning: No rollout buffer available to check rewards")
                except Exception as e:
                    print(f"Error accessing rollout buffer: {e}")
                    if self.verbose > 0:
                        print("Warning: Could not find any reward statistics")

        return True