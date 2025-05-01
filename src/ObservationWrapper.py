import cv2
import gymnasium 
import numpy as np

# Returns an observation containing a resised image and other info.
# frame_skip=the number of frames to skip between actions to speed up training
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape, frame_skip):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = frame_skip 

        # create new observation space with the new shape
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        if observation.shape[-1] != 3:  # if the channels are not 3 (corresponding to RGB)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) # convert to RGB24
        return observation
