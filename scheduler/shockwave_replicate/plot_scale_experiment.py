import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

root_dir = os.path.dirname(os.path.realpath(__file__))


def load_data(pickle_dir, metrics_to_plot):
    data = {}
    # Iterate over all pickle files in the directory
    for filename in os.listdir(pickle_dir):
        if filename.endswith(".pickle"):
            filepath = os.path.join(pickle_dir, filename)
            with open(filepath, "rb") as file:
                pickle_object = pickle.load(file)
                num_gpus = int(pickle_object["num_gpus"])
                policy = pickle_object["policy"]
                # Initialize dictionary structure
                if num_gpus not in data:
                    data[num_gpus] = {}
                if policy not in data[num_gpus]:
                    data[num_gpus][policy] = {}
                # Store data for each metric
                for metric in metrics_to_plot:
                    if metric in pickle_object:
                        if metric not in data[num_gpus][policy]:
                            data[num_gpus][policy][metric] = []
                        data[num_gpus][policy][metric].append(pickle_object[metric])
                        # data[num_gpus][policy][metric] = pickle_object[metric]
    return data


def plot_data(data, metrics_to_plot, save_dir, metrics_names):
    num_gpus_list = sorted(data.keys())
    policies = sorted(set(policy for gpu in data.values() for policy in gpu.keys()))
    colors = ["black", "grey", "red"]  # Ensure you have enough colors for all policies

    # Set up the figure and axes for the subplots
    fig, axes = plt.subplots(
        1, len(metrics_to_plot), figsize=(15, 5)
    )  # One row of plots

    # Define the width of each bar and the separation between groups of bars
    bar_width = 0.8 / len(policies)  # Adjust bar width based on number of policies
    offset = np.arange(len(num_gpus_list))  # Initial positions for each num_gpus

    # Create plots
    for j, metric in enumerate(metrics_to_plot):
        ax = axes[j]

        # Create a group of bar plots for each num_gpus
        for idx, num_gpus in enumerate(num_gpus_list):
            # Initialize a list to hold the positions for this group
            group_positions = offset[idx] + np.arange(len(policies)) * bar_width

            # Plot a bar for each policy value
            for k, policy in enumerate(policies):
                values = (
                    np.mean(data[num_gpus][policy][metric])
                    if metric in data[num_gpus][policy]
                    else 0
                )
                ax.barh(
                    group_positions[k],
                    values,
                    color=colors[k % len(colors)],  # Cycle through colors
                    height=bar_width,
                    align="center",
                    edgecolor="black",
                )

            # Move the starting position of the next group so there's space between groups
            offset[idx] += len(policies) * bar_width

        # Set labels, titles, ticks
        ax.set_xlabel(metrics_names[metric], fontsize=12, labelpad=10)
        tick_positions = offset + (len(policies) * bar_width / 2) - (bar_width / 2)
        ax.set_yticks(
            tick_positions
        )  # Adjust y-ticks to be in the middle of each group
        # ax.set_yticks(offset - (len(policies) * bar_width / 2))  # Adjust y-ticks to be in the middle of groups
        ax.set_yticklabels(num_gpus_list)
        ytick_labels = [f"{num} GPUs" for num in num_gpus_list]  ###
        ax.set_yticklabels(ytick_labels)
        ax.invert_yaxis()  # Invert to have the first entry at the top
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add the legend centrally above all subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in colors]
    labels = policies
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(policies),
        frameon=True,
        fontsize="large",
        bbox_to_anchor=(0.5, 0.98),
    )
    legend.get_frame().set_edgecolor("grey")
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.8)  # Adjust alpha for a soft look
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(save_dir, "gpu_metrics_plots.png"))


if __name__ == "__main__":
    pickle_dir = os.path.join(root_dir, "results/pickle")
    metrics_to_plot = ["makespan", "avg_jct"]
    data = load_data(pickle_dir, metrics_to_plot)
    print(data)
    metrics_names = {
        "makespan": "Makespan (s)",
        "avg_jct": "Average Job Completion Time (s)",
    }
    save_dir = os.path.join(root_dir, "results/plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_data(data, metrics_to_plot, save_dir, metrics_names)
