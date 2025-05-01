import os
import optuna
from DRL_Agent import DRL_Agent
from ResultsLogger import append_rows

RESULTS_DIR = './results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

class OptunaOptimiser:
    def __init__(self, environment_id, learning_alg, n_trials=50):
        self.environment_id = environment_id
        self.learning_alg = learning_alg
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = -float('inf')
        self.baseline_value = None
        self.results_file = os.path.join(RESULTS_DIR, f"optuna_results_{learning_alg}_{environment_id}.txt")

    def objective(self, trial):
        # Suggest values for hyperparameters
        hyperparams = {
            "l_rate": trial.suggest_float("l_rate", 1e-5, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "n_steps": trial.suggest_int("n_steps", 16, 2048),
            "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128, 64]]),
            "exploration_initial_eps": trial.suggest_float("eps_start", 0.9, 1.0),
            "exploration_final_eps": trial.suggest_float("eps_end", 0.01, 0.1),
        }
        
        # algo-specific parameters
        if self.learning_alg == "DQN":
            hyperparams.update({
                "buffer_size": trial.suggest_int("buffer_size", 1000, 50000),
                "batch_size": trial.suggest_int("batch_size", 32, 256),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 1.0),
            })
        
        agent = DRL_Agent(
            self.environment_id,
            self.learning_alg,
            train_mode=True,
            seed=None,
            hyperparams=hyperparams
        )
        
        print(f"\nTrial {trial.number}: Using hyperparameters:")
        for param, value in hyperparams.items():
            print(f"  {param}: {value}")
        
        try:
            mean_reward, *_ = agent.run()
            
            print(f"Trial {trial.number} finished with reward: {mean_reward}")
            
            with open(self.results_file, "a") as f:
                f.write(f"Trial {trial.number}:\n")
                for param, value in hyperparams.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"  Reward: {mean_reward}\n\n")
            
            # Update best parameters if this is the best run
            if mean_reward > self.best_value:
                self.best_value = mean_reward
                self.best_params = hyperparams
                # Save the best model so far
                best_model_path = os.path.join(agent.policy_dir, f"best_{self.learning_alg}_{self.environment_id}.policy.pkl")
                if os.path.exists(agent.policy_filename):
                    os.rename(agent.policy_filename, best_model_path)
            append_rows([{
                "env": self.environment_id,
                "algorithm": self.learning_alg,
                "trial": trial.number,
                **hyperparams,
                "mean_reward": mean_reward
            }], csv_path=os.path.join(RESULTS_DIR, "optuna_trials.csv"))
            return mean_reward
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            return -float('inf')

    def run_baseline(self):
        print("\nRunning baseline with default parameters...")
        agent = DRL_Agent(self.environment_id, self.learning_alg, train_mode=True)
        self.baseline_value, *_ = agent.run()
        print(f"Baseline performance: {self.baseline_value}")
        
        with open(self.results_file, "a") as f:
            f.write("Baseline performance:\n")
            f.write(f"  Reward: {self.baseline_value}\n\n")
        
        return self.baseline_value

    def optimise(self):
        with open(self.results_file, "w") as f:
            f.write(f"Optuna Optimisation for {self.learning_alg} on {self.environment_id}\n")
            f.write("===========================================\n\n")
        
        self.run_baseline()
        
        print(f"\nStarting optimisation with {self.n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print("\nOptimisation completed!")
        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)
        
        improvement = ((study.best_value - self.baseline_value) / abs(self.baseline_value)) * 100
        print(f"Improvement over baseline: {improvement:.2f}%")
        
        with open(self.results_file, "a") as f:
            f.write("Final Results:\n")
            f.write(f"Best parameters: {study.best_params}\n")
            f.write(f"Best value: {study.best_value}\n")
            f.write(f"Improvement over baseline: {improvement:.2f}%\n")
        
        append_rows([{
            "env": self.environment_id,
            "algorithm": self.learning_alg,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "baseline": self.baseline_value
        }], csv_path=os.path.join(RESULTS_DIR, "optuna_best.csv"))
        return study.best_params, study.best_value
