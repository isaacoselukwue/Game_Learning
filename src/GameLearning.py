import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd, glob, os
import random
import re
import vizdoom.gymnasium_wrapper

from DRL_Agent import DRL_Agent
from OptunaOptimiser import OptunaOptimiser
from ResultsLogger import append_rows

RESULTS_DIR = './results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def check_for_policies(environment_id, learning_algs):
    """Check if trained policies exist for the given environment and algorithms"""
    policy_dir = './policies'
    available_policies = {}
    for alg in learning_algs:
        policies = [f for f in os.listdir(policy_dir) if (f.startswith(f"{alg}-{environment_id}-") and f.endswith(".policy.pkl"))]
        if policies:
            available_policies[alg] = policies
    
    return available_policies

def train_multiple_seeds(environment_id, learning_alg, encoder_type, seeds, hyperparams=None):
    print(f"\nTraining {learning_alg} with {encoder_type} encoder using {len(seeds)} seeds: {seeds}")
    
    results = []
    for seed in seeds:
        print(f"\n--- Training with seed {seed} ---")
        agent = DRL_Agent(
            environment_id, 
            learning_alg, 
            train_mode=True, 
            seed=seed,
            hyperparams=hyperparams,
            encoder_type=encoder_type
        )
        m_reward, s_reward, m_steps, t_time = agent.run()
        results.append(dict(seed=seed, mean_reward=m_reward, std_reward=s_reward, avg_steps=m_steps, train_time=t_time))
        print(f"Seed {seed} → reward {m_reward:.2f}  steps {m_steps:.1f}  time {t_time:.1f}s")
    
    avg_reward = np.mean([r["mean_reward"] for r in results])
    std_reward = np.std([r["mean_reward"] for r in results])
    avg_steps  = np.mean([r["avg_steps"]   for r in results])
    train_time = np.sum([r["train_time"]  for r in results])
    
    print(f"\n--- Results Summary for {encoder_type} encoder ---")
    print(f"Seeds used: {seeds}")
    print(f"Individual rewards: {results}")
    print(f"Average reward: {avg_reward}")
    print(f"Standard deviation: {std_reward}")
    
    result_file = os.path.join(RESULTS_DIR, f"results_{learning_alg}_{encoder_type}_{'-'.join(map(str, seeds))}.txt")
    with open(result_file, "w") as f:
        f.write(f"Results for {learning_alg} with {encoder_type} encoder\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Individual rewards: {results}\n")
        f.write(f"Average reward: {avg_reward}\n")
        f.write(f"Standard deviation: {std_reward}\n")
    

    append_rows([{
        "phase": "train",
        "env": environment_id,
        "algorithm": learning_alg,
        "encoder": encoder_type,
        "seeds": "-".join(map(str, seeds)),
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_steps": avg_steps,
        "train_time_s": train_time
    }], csv_path=os.path.join(RESULTS_DIR, "train_summary.csv"))

    
    monitor_dirs = [f'./logs/{learning_alg}_{seed}' for seed in seeds]
    curve_dfs = []
    for d in monitor_dirs:
        try:
            df = pd.read_csv(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)[0],
                             skiprows=1)
            df["rolling_reward"] = df["r"].rolling(50, min_periods=1).mean()
            curve_dfs.append(df[["t", "rolling_reward"]])
        except Exception:
            pass

    if curve_dfs:
        merged = pd.concat(curve_dfs).groupby("t").mean().reset_index()
        plt.figure(figsize=(10, 6))
        plt.plot(merged["t"], merged["rolling_reward"])
        plt.title(f"{learning_alg} ({encoder_type}) – reward curve (avg of {len(seeds)} seeds)")
        plt.xlabel("Environment steps")
        plt.ylabel("Reward (50-ep MA)")
        plt.savefig(os.path.join(RESULTS_DIR, f"results_{learning_alg}_{encoder_type}_curve.png"))

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(seeds)), [r["mean_reward"] for r in results], tick_label=[f"Seed {s}" for s in seeds])
    plt.axhline(y=avg_reward, color='r', linestyle='-', label=f'Average: {avg_reward:.2f}')
    plt.legend()
    plt.title(f"{learning_alg} with {encoder_type} encoder – Rewards by Seed")
    plt.ylabel("Mean Reward")
    plt.savefig(os.path.join(RESULTS_DIR, f"results_{learning_alg}_{encoder_type}.png"))
    
    return avg_reward, std_reward, results

def main_menu():
    """Interactive menu for the user"""
    print("\n===== REINFORCEMENT LEARNING TRAINING/TESTING SYSTEM =====")
    
    environments = {
        "1": "LunarLander-v3",
        "2": "VizdoomTakeCover-v0",
        "3": "VizdoomPredictPosition-v0",
        "4": "VizdoomHealthGathering-v0"
    }
    learning_algorithms = ["DQN", "A2C", "PPO"]
    encoder_types = ["cnn", "transformer", "hybrid"]
    
    print("\nSelect environment:")
    for key, env in environments.items():
        print(f"{key}: {env}")
    
    env_choice = input("\nEnter your choice: ")
    while env_choice not in environments:
        print("Invalid choice. Please try again.")
        env_choice = input("Enter your choice: ")
    
    environment_id = environments[env_choice]
    
    available_policies = check_for_policies(environment_id, learning_algorithms)
    
    print("\nSelect action:")
    print("1: Train a new agent")
    print("2: Train multiple agents with different seeds")
    print("3: Optimise hyperparameters with Optuna")
    
    if available_policies:
        print("4: Test an existing agent")
    else:
        print("(Testing is not available - no trained policies found)")
    
    action_choice = input("\nEnter your choice: ")
    while not (action_choice in ["1", "2", "3"] or (action_choice == "4" and available_policies)):
        print("Invalid choice. Please try again.")
        action_choice = input("Enter your choice: ")
    
    # Training or optimisation
    if action_choice in ["1", "2", "3"]:
        print("\nSelect learning algorithm:")
        for i, alg in enumerate(learning_algorithms, 1):
            print(f"{i}: {alg}")
        
        alg_choice = input("\nEnter your choice: ")
        while not alg_choice.isdigit() or int(alg_choice) < 1 or int(alg_choice) > len(learning_algorithms):
            print("Invalid choice. Please try again.")
            alg_choice = input("Enter your choice: ")
        
        learning_alg = learning_algorithms[int(alg_choice) - 1]
        
        if environment_id.find("Vizdoom") > -1:
            print("\nSelect encoder type:")
            for i, encoder in enumerate(encoder_types, 1):
                print(f"{i}: {encoder}")
            
            encoder_choice = input("\nEnter your choice: ")
            while not encoder_choice.isdigit() or int(encoder_choice) < 1 or int(encoder_choice) > len(encoder_types):
                print("Invalid choice. Please try again.")
                encoder_choice = input("Enter your choice: ")
            
            encoder_type = encoder_types[int(encoder_choice) - 1]
        else:
            encoder_type = "cnn"
        
        if action_choice == "1":  # Train single agent
            seed = input("\nEnter seed value (or leave blank for random seed): ")
            if seed.isdigit():
                seed = int(seed)
            else:
                seed = None
                
            agent = DRL_Agent(environment_id, learning_alg, train_mode=True, seed=seed, encoder_type=encoder_type)
            agent.run()
            print("\nTraining completed!")
            
        elif action_choice == "2":  # Train with multiple seeds
            print("\nTraining with multiple seeds")
            seed_input = input("\nEnter seeds separated by commas (at least 3 recommended, e.g., 42,123,456): ")
            try:
                seeds = [int(s.strip()) for s in seed_input.split(",")]
                if len(seeds) < 3:
                    print("Warning: It's recommended to use at least 3 seeds for reliable results.")
                    confirm = input("Continue with fewer than 3 seeds? (y/n): ")
                    if confirm.lower() != 'y':
                        return
            except ValueError:
                print("Invalid seed format. Using 3 random seeds instead.")
                seeds = [random.randint(0, 1000) for _ in range(3)]
                print(f"Random seeds generated: {seeds}")
            
            train_multiple_seeds(environment_id, learning_alg, encoder_type, seeds)
            print("\nMulti-seed training completed!")
            
        elif action_choice == "3":  # Optimise
            n_trials = input("\nEnter number of optimisation trials (default: 10): ")
            n_trials = int(n_trials) if n_trials.isdigit() and int(n_trials) > 0 else 10
            
            optimiser = OptunaOptimiser(environment_id, learning_alg, n_trials)
            best_params, best_value = optimiser.optimise()
            print("\nOptimisation completed!")
            
            train_best = input("\nDo you want to train a model with the best parameters? (y/n): ")
            if train_best.lower() == 'y':
                agent = DRL_Agent(environment_id, learning_alg, train_mode=True, hyperparams=best_params, encoder_type=encoder_type)
                agent.run()
                print("\nTraining with optimised parameters completed!")
    
    # Testing
    elif action_choice == "4":
        # Find available encoder types in policies
        policies_by_alg = {}
        for alg in learning_algorithms:
            for alg in learning_algorithms:
                policies = [f for f in os.listdir('./policies') if 
                        (f.startswith(f"{alg}-{environment_id}") and f.endswith(".policy.pkl"))]
                if policies:
                    policies_by_alg[alg] = policies
        
        print("\nSelect learning algorithm for testing:")
        algs_with_policies = list(policies_by_alg.keys())
        if not algs_with_policies:
            print("No policies found for any algorithm.")
            return
            
        for i, alg in enumerate(algs_with_policies, 1):
            print(f"{i}: {alg}")
        
        alg_choice = input("\nEnter your choice: ")
        while not alg_choice.isdigit() or int(alg_choice) < 1 or int(alg_choice) > len(algs_with_policies):
            print("Invalid choice. Please try again.")
            alg_choice = input("Enter your choice: ")
        
        learning_alg = algs_with_policies[int(alg_choice) - 1]
        
        matching_policies = policies_by_alg[learning_alg]
        
        encoder_type = "cnn"
        policy_file = matching_policies[0]
        
        encoder_match = re.search(f"{learning_alg}-{environment_id}-(.+?)-seed", policy_file)
        if encoder_match:
            encoder_type = encoder_match.group(1)
            
        print("\nSelect policy for testing:")
        for i, policy in enumerate(matching_policies, 1):
            print(f"{i}: {policy}")
        
        policy_choice = input("\nEnter your choice: ")
        while not policy_choice.isdigit() or int(policy_choice) < 1 or int(policy_choice) > len(matching_policies):
            print("Invalid choice. Please try again.")
            policy_choice = input("Enter your choice: ")
        
        policy_file = matching_policies[int(policy_choice) - 1]
        seed_match = re.search(r'seed(\d+)', policy_file)
        seed = int(seed_match.group(1)) if seed_match else 0
        
        encoder_match = re.search(f"{learning_alg}-{environment_id}-(.+?)-seed", policy_file)
        selected_encoder = encoder_match.group(1) if encoder_match else "cnn"
        
        agent = DRL_Agent(environment_id, learning_alg, train_mode=False, seed=seed, encoder_type=selected_encoder)
        agent.policy_filename = os.path.join('./policies', policy_file)
        agent.run()
        print("\nTesting completed!")


def load_monitor(log_dir):
    df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
    return (df["l"].mean(), df["r"].mean(), df["t"].iloc[-1])


def check_for_policies(environment_id, learning_algs):
    """Check if trained policies exist for the given environment and algorithms"""
    policy_dir = './policies'
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
    available_policies = {}
    for alg in learning_algs:
        policies = [f for f in os.listdir(policy_dir) if f.startswith(f"{alg}-{environment_id}") and f.endswith(".policy.pkl")]
        if policies:
            available_policies[alg] = policies
    
    return available_policies


if __name__ == "__main__":
    main_menu()