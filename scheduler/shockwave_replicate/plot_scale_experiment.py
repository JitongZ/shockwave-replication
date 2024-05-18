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
                num_gpus = pickle_object["num_gpus"]
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
    colors = ["black", "grey"]  # Custom color scheme

    # Set up the figure and axes for the subplots
    fig, axes = plt.subplots(
        1, len(metrics_to_plot), figsize=(15, 5)
    )  # One row of plots

    # Create plots
    for j, metric in enumerate(metrics_to_plot):
        ax = axes[j]
        bar_positions = np.arange(len(policies))
        # Gather data for each policy and metric
        for idx, policy in enumerate(policies):
            values = (
                np.mean(data[num_gpus_list[0]][policy][metric])
                if metric in data[num_gpus_list[0]][policy]
                else 0
            )
            ax.barh(
                bar_positions[idx],
                values,
                color=colors[idx],
                height=0.01,
                align="center",
                edgecolor="black",
            )

        # Set labels, titles, ticks
        # ax.set_xticks([])
        ax.set_xlabel(metrics_names[metric], fontsize=12, labelpad=10)
        ax.set_yticks([])
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
