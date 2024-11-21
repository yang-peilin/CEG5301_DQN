import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set backend to 'Agg' to avoid GUI-related issues
matplotlib.use('Agg')


def plot_mean_ep_return(result_path):
    # Load the mean episode returns from the results file
    if os.path.isfile(result_path):
        with open(result_path, 'rb') as file:
            mean_ep_returns = pickle.load(file)
        used_steps = np.arange(1, len(mean_ep_returns) + 1) * int(1e3)  # Assuming save frequency is 1000 steps

        # Plotting the figure
        plt.figure(figsize=(10, 5))
        plt.plot(used_steps, mean_ep_returns, label='Mean Return over 100 Episodes')
        plt.xlabel('Used Steps')
        plt.ylabel('Mean Return (Last 100 Episodes)')
        plt.title('DQN Training: Mean Return over 100 Episodes vs. Used Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(result_path), 'mean_return_plot.png'))
        print(f"Plot saved as 'mean_return_plot.png' in {os.path.dirname(result_path)}.")
    else:
        print(f"Result file '{result_path}' not found.")


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    # result_path = os.path.join(current_path, 'plots/dqn_result_o_Pendulum.pkl')  # Replace with your actual result file path
    result_path = "D:\\桌面\\NUS\\5301\\HOMEWORK 5\\CEG5301_DQN\\DQN\\data\\plots\\dqn_result_o_Pendulum.pkl"
    plot_mean_ep_return(result_path)
