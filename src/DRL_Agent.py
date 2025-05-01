import gymnasium
import pandas as pd, glob, os
import pickle
import random
import sys
import time
from ObservationWrapper import ObservationWrapper
from GrayscaleObservationWrapper import GrayscaleObservationWrapper
from HybridCNN_TransformerExtractor import HybridCNN_TransformerExtractor
from ResultsLogger import append_rows
from TransformerFeatureExtractor import TransformerFeatureExtractor
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecMonitor

RESULTS_DIR = './results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Class for creating DRL agents through environment setup, model creation, training, evaluation, and policy rendering
class DRL_Agent:
    def __init__(self, environment_id, learning_alg, train_mode=True, seed=None, n_envs=8, frame_skip=4, hyperparams=None, encoder_type="cnn"):
        self.environment_id = environment_id
        self.learning_alg = learning_alg
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)
        if encoder_type and encoder_type != "cnn":
            self.policy_filename = f"{learning_alg}-{environment_id}-{encoder_type}-seed{self.seed}.policy.pkl"
        else:
            self.policy_filename = f"{learning_alg}-{environment_id}-seed{self.seed}.policy.pkl"
        self.n_envs = n_envs if train_mode else 1  # number of environments for training or 1 for testing
        self.frame_skip = frame_skip # number of frames to skip between actions in the environment
        self.image_shape = (84, 84) # image res (height, width):e.g., (240, 320); (120, 160); (60, 80); 
        self.training_timesteps = 100 # total number of timesteps for training the agent
        self.num_test_episodes = 20 # number of episodes to run for testing the trained agent
        self.encoder_type = encoder_type # type of encoder to use for feature extraction (CNN, Transformer, or Hybrid)
        
        # Default hyperparameters - will be overridden if hyperparams is provided
        self.l_rate = 0.00083 # learning rate for the optimiser during training
        self.gamma = 0.995 # discount factor for future rewards (used in RL algorithms)
        self.n_steps = 512 # number of steps/actions the agent will take before updating the model
        self.buffer_size = 10000  # replay buffer size for DQN
        self.batch_size = 64  # batch size for DQN
        self.exploration_fraction = 0.9  # exploration fraction for DQN
        
        # Override defaults with provided hyperparameters if any
        if hyperparams:
            for key, value in hyperparams.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        self.policy_rendering = True # if True, shows visualisations of the learnt behaviour
        self.rendering_delay = 0.05 if self.environment_id.find("Vizdoom") > 0 else 0 # delay in rendering
        self.log_dir = f'./logs/{learning_alg}_{self.seed}' # directory to store the logs containing agent performance
        self.policy_dir = './policies' # directory to store the trained policies
        if not os.path.exists(self.policy_dir):
            os.makedirs(self.policy_dir)
        if encoder_type and encoder_type != "cnn":
            self.policy_filename = os.path.join(self.policy_dir, f"{learning_alg}-{environment_id}-{encoder_type}-seed{self.seed}.policy.pkl")
        else:
            self.policy_filename = os.path.join(self.policy_dir, f"{learning_alg}-{environment_id}-seed{self.seed}.policy.pkl")
        self.model = None # initialises the model that will define the agent's policy & learning behavior
        self.policy = None # initialises the policy to "MlpPolicy" or "CnnPolicy" depending on the environment
        self.environment = None # initialises the environment as None, to be set later (gym environment)

        self._check_environment()
        self._create_log_directory()

    def _check_environment(self):
        available_envsA = [env for env in gymnasium.envs.registry.keys() if "LunarLander" in env]
        available_envsB = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
        if self.environment_id in available_envsA :
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsA}")
        elif self.environment_id in available_envsB:
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsB}")
        else:
            print(f"UNKNOWN environment={self.environment_id}")
            print(f"AVAILABLE_ENVS={available_envsA, available_envsB}")
            sys.exit(0)

    # creates a log directory if it doesn't exist already
    def _create_log_directory(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir) 
            print(f"Log directory created: {self.log_dir}")
        else:
            print(f"Log directory {self.log_dir} already exists!")
    
    # wrapper function to customise environments for image-based obsertations
    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        env = GrayscaleObservationWrapper(env) # convert to grayscale
        if self.train_mode: 
            # scale rewards for training stability, only during training
            env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01) 
        return env

    # Create the vectorised environment with multiple parallel environments.
    def create_environment(self, use_rendering=False):
        print("self.environment_id="+str(self.environment_id))

        if self.environment_id.find("Vizdoom") == -1:
            if use_rendering: 
                self.environment = gymnasium.make(self.environment_id, render_mode="human")
            else:
                self.environment = gymnasium.make(self.environment_id)
            self.environment = DummyVecEnv([lambda: self.environment])
            self.environment = VecMonitor(self.environment, self.log_dir) 
            self.policy = "MlpPolicy"
        
        else:
            self.environment = make_vec_env(
                self.environment_id,
                n_envs=self.n_envs,
                seed=self.seed,
                monitor_dir=self.log_dir,
                wrapper_class=self.wrap_env  # applies wrappers inside this function
            )
            self.environment = VecFrameStack(self.environment, n_stack=4)  # stacks frames for temporal context
            self.environment = VecTransposeImage(self.environment) # transposes image for correct format (channel first)
            self.policy = "CnnPolicy"

        print("self.environment.action_space:", self.environment.action_space)

    def create_model(self):
        policy_kwargs = {}
        if getattr(self, "net_arch", None):
            policy_kwargs["net_arch"] = self.net_arch
        if self.environment_id.find("Vizdoom") > -1:
            if self.encoder_type == "transformer":
                self.policy = "CnnPolicy"
                policy_kwargs = dict(
                    features_extractor_class=TransformerFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=256)
                )
            elif self.encoder_type == "hybrid":
                self.policy = "CnnPolicy"
                policy_kwargs = dict(
                    features_extractor_class=HybridCNN_TransformerExtractor,
                    features_extractor_kwargs=dict(features_dim=256, cnn_features=128)
                )
            else:
                self.policy = "CnnPolicy"
        
        if self.learning_alg == "DQN":
            self.model = DQN(
                self.policy, 
                self.environment, 
                seed=self.seed, 
                learning_rate=self.l_rate, 
                gamma=self.gamma, 
                buffer_size=self.buffer_size, 
                batch_size=self.batch_size, 
                exploration_fraction=self.exploration_fraction, 
                exploration_initial_eps= getattr(self, "exploration_initial_eps", 1.0),
                exploration_final_eps= getattr(self, "exploration_final_eps", 0.05),
                verbose=1,
                policy_kwargs=policy_kwargs
            )
            
        elif self.learning_alg == "A2C":
            self.model = A2C(
                self.policy, 
                self.environment, 
                seed=self.seed, 
                learning_rate=self.l_rate, 
                gamma=self.gamma, 
                n_steps=self.n_steps,
                verbose=1,
                policy_kwargs=policy_kwargs
            )
            
        elif self.learning_alg == "PPO":
            self.model = PPO(
                self.policy, 
                self.environment, 
                seed=self.seed, 
                learning_rate=self.l_rate, 
                gamma=self.gamma, 
                n_steps=self.n_steps,
                verbose=1,
                policy_kwargs=policy_kwargs
            )
            
        else:
            print(f"Unknown LEARNING_ALG={self.learning_alg}")
            sys.exit(0)

    def train_or_load_model(self):
        print(self.model)
        if self.train_mode:
            start_time = time.time()
            self.model.learn(total_timesteps=self.training_timesteps)
            self.training_time = time.time() - start_time
            print(f"Training completed in {self.training_time:.2f} seconds")
            print(f"Saving policy {self.policy_filename}")
            pickle.dump(self.model.policy, open(self.policy_filename, 'wb'))
        else:
            print("Loading policy...")
            with open(self.policy_filename, "rb") as f:
                policy = pickle.load(f)
            self.model.policy = policy

    def evaluate_policy(self):
        print("Evaluating policy...")
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=self.num_test_episodes)
        monitor_files = glob.glob(os.path.join(self.log_dir, "**", "*.csv"), recursive=True)
        avg_steps = None
        last_wall_time = None
        if monitor_files:
            df = pd.concat([pd.read_csv(f,skiprows=1) for f in monitor_files])
            avg_steps = df["l"].mean()
            last_wall_time = df["t"].iloc[-1]

        print(f"EVALUATION: mean_reward={mean_reward} std_reward={std_reward}, reward={mean_reward:.2f}±{std_reward:.2f}, "
              f"avg_steps={avg_steps:.1f}, elapsed_s={last_wall_time:.1f}")
        return mean_reward, std_reward, avg_steps, last_wall_time

    # render the agent's behavior in the environment and track cumulative reward
    def render_policy(self):
        steps_per_episode = 0
        reward_per_episode = 0
        total_cummulative_reward = 0
        total_steps = 0
        episode = 1
        self.create_environment(True)
        env = self.environment
        obs = env.reset()

        print("DEMONSTRATION EPISODES:")
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            # ensemble = EnsemblePolicy([self.model, self.model,self.model]) #uncomment for ensemble policy usage
            # action, _states = ensemble.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps_per_episode += 1
            reward_per_episode += reward
            if any(done):
                print(f"episode={episode}, steps_per_episode={steps_per_episode}, reward_per_episode={reward_per_episode}")
                total_cummulative_reward += reward_per_episode
                total_steps += steps_per_episode
                steps_per_episode = 0
                reward_per_episode = 0
                episode += 1
                obs = env.reset()
            if self.policy_rendering:
                env.render("human")
                time.sleep(self.rendering_delay)
            if episode > self.num_test_episodes:
                print(f"total_cummulative_reward={total_cummulative_reward} avg_cummulative_reward={total_cummulative_reward / self.num_test_episodes}")
                monitor_files = glob.glob(os.path.join(self.log_dir, "**", "*.csv"), recursive=True)
                test_time_s = None
                if monitor_files:
                    df = pd.concat([pd.read_csv(f, skiprows=1) for f in monitor_files])
                    test_time_s = df["t"].iloc[-1]
                append_rows([{
                    "phase": "test",
                    "env": self.environment_id,
                    "algorithm": self.learning_alg,
                    "encoder": self.encoder_type,
                    "seed": self.seed,
                    "avg_reward": total_cummulative_reward / self.num_test_episodes,
                    "avg_steps": total_steps / self.num_test_episodes,
                    "test_time_s": test_time_s
                }], csv_path=os.path.join(RESULTS_DIR, "test_summary.csv"))
                break
        env.close()

    # main method to run the DRL agent
    def run(self):
        self.create_environment()
        self.create_model()
        self.train_or_load_model()
        mean_reward, std_reward, avg_steps, last_wall_time = self.evaluate_policy()
        
        if self.train_mode:
            return mean_reward, std_reward, avg_steps, last_wall_time
        else:
            self.render_policy()
            return mean_reward, std_reward, avg_steps, last_wall_time
