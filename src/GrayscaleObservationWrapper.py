import cv2
import gymnasium
import numpy as np
# Converts RGB observations to grayscale to reduce input dimensionality.
# gymnasium.spaces.Box() creates a continuous space for observations.
# see https://gymnasium.farama.org/main/_modules/gymnasium/spaces/box/
class GrayscaleObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        ) # only 1 channel (grayscale) instead of 3 (RGB)

    def observation(self, observation):
        # check that the observation is in RGB format before converting to grayscale
        if observation.shape[-1] != 3:  # if the channels are not 3 (corresponding to RGB)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) # convert to RGB24

        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # convert to grayscale
        gray = np.expand_dims(gray, axis=-1) # expand dimensions to match input shape (H, W, 1)
        return gray
