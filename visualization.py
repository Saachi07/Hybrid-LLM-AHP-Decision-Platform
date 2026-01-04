import matplotlib.pyplot as plt
import numpy as np

def plot_ci_trends(results_dict, title="Consistency Index (CI) over 10 Rounds"):
    """
    Plots CI values across rounds (Line Chart).
    results_dict structure: {'ChatGPT': [ci1, ci2...], 'DeepSeek': [...]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    markers = ['o', 's', '^', 'D', 'x', '*']
    for i, (llm, values) in enumerate(results_dict.items()):
        marker = markers[i % len(markers)]
        ax.plot(range(1, 11), values, marker=marker, label=llm, linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("CI Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def plot_importance_counts(counts_data, title, criteria_names):
    """
    Bar chart for Most/Least Important criteria counts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(criteria_names))
    width = 0.15
    
    # counts_data: {'ChatGPT': [count_c1, count_c2...], ...}
    multiplier = 0
    for llm, counts in counts_data.items():
        offset = width * multiplier
        ax.bar(x + offset, counts, width, label=llm)
        multiplier += 1

    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(x + width, criteria_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_weights_for_sites(site_weights, criteria_name, llm_name):
    """
    Scatter/Dot plot for site weights (replicating graphs.py style).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sites = [f"S{i+1}" for i in range(10)]
    
    # site_weights is a list of lists (10 rounds x 10 sites)
    rounds = len(site_weights)
    colors = plt.cm.viridis(np.linspace(0, 1, rounds))
    
    for r in range(rounds):
        ax.scatter(sites, site_weights[r], label=f"Round {r+1}", alpha=0.7)
        
    ax.set_title(f"Weights of {llm_name} under {criteria_name}")
    ax.set_ylabel("Weight Value")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig