import pandas as pd
import matplotlib.pyplot as plt

def plot_smoothed_average_return(file_path, window_size=10, label=None):
    """
    Reads a CSV file, extracts the 'AverageReturn' column, applies smoothing,
    and plots the result.

    Parameters:
    - file_path (str): Path to the CSV file.
    - window_size (int): Window size for moving average smoothing.
    - label (str): Optional label for the plot.
    """
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Check if the column exists
    if "NormReturn" not in df.columns:
        raise ValueError(f"'NormReturn' column not found in {file_path}")
    
    # Apply smoothing
    data = df["NormReturn"][:] 
    smoothed = data.rolling(window=window_size, min_periods=1).mean()
    
    # Plot
    plt.plot(smoothed, label=label or file_path)

# Example usage for multiple files:
def plot_multiple_returns(base_path, env_name, file_paths, window_size=10):
    """
    Plot multiple smoothed NormReturn curves from a list of CSV files.

    Parameters:
    - file_paths (list): List of CSV file paths.
    - window_size (int): Window size for moving average smoothing.
    """
    plt.figure(figsize=(10, 6))
    for path in file_paths:
        label = path
        file_name = env_name + "/progress.csv"
        plot_smoothed_average_return(base_path + path + file_name, window_size, label)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.title(env_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# maze2d-medium-v1

base_path = "./results/"
file_list = [
    # 1000： use target_q, 4000: use q_min, 1200: use target_q, update 2. 1300: use target_q, update 1
            # "Exp0309/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi_target/Exp0qtarget_w/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi_target/Exp0qtarget_n/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi_target/Exp0qmax_w/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi_target/Exp0qmax_n/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi_target/Exp0qmin_w/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi_target/Exp0qmin_n/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi/Exp0qmax_w/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi/Exp0qmax_n/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi/Exp0qtarget_w/",     # qtarget-qmin, use q_max for bootstrap
            # "q_pi/Exp0qtarget_n/",     # qtarget-qmin, use q_max for bootstrap
            # "Exp9999/",     # qtarget-qmin, use q_max for bootstrap
            # "Exp9997/",     # qtarget-qmin, use q_max for bootstrap
            "Exp0700/",     # qtarget-qmin, use q_max for bootstrap
            "Exp0800/",     # qtarget-qmin, use q_max for bootstrap
            # "Exp0703/",     # qtarget-qmin, use q_max for bootstrap
            # "Exp0802/",     # qtarget-qmin, use q_max for bootstrap
            # "Exp0100/",     # qtarget-qmin, use q_max for bootstrap
             ]

# 1000 no range 2.0, center 0.5
# 100: tau 0.003, min pi, 200: tao 0.003 no min pi
# 300: small lr

# # env_name = "antmaze-umaze-diverse-v2"
# env_name = "antmaze-large-diverse-v2"
# # env_name = "maze2d-medium-v1"

env_list = [
            # "antmaze-umaze-diverse-v2",
            # "antmaze-medium-diverse-v2",
            "antmaze-large-diverse-v2",
            # "antmaze-large-play-v2",
            # "maze2d-medium-v1",
            # "maze2d-umaze-v1",
            "maze2d-large-v1",
            # "hopper-random-v2",
            # "hopper-medium-v2",
            # "hopper-expert-v2",
            # "walker2d-random-v2",
            # "walker2d-medium-v2",
            # "walker2d-expert-v2",
            # "halfcheetah-random-v2",
            ]

for env_name in env_list:
    try:
        plot_multiple_returns(base_path, env_name, file_list, window_size=3)
    except Exception as e:
        print(f"Error plotting {env_name}: {e}")
