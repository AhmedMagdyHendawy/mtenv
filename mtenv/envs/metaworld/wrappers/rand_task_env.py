import gym
import gym.spaces
import gym.spaces.utils
import numpy as np
import random

class RandEnvWrapper(gym.Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
        self,
        env,
        task_list=None,
    ):
        super().__init__(env)

        self._task_list = task_list

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        if self._task_list is not None:
            task = random.choice(self._task_list)
            self.env.set_task(task)
            
        ret = self.env.reset(**kwargs)
        
        return ret