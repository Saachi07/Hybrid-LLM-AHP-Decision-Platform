import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Shared style constants ─────────────────────────────────────────────────────
MARKERS   = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,2)),
              (0,(1,1)), (0,(3,5,1,5)), '-', '--']
HATCHES   = ['/', '\\', 'x', '-', '+', '|', 'o', 'O', '.', '*']

# Grayscale shades for lines/bars (dark → light)
GRAYS = ['#000000', '#1a1a1a', '#333333', '#4d4d4d',
         '#666666', '#808080', '#999999', '#b3b3b3', '#cccccc', '#e6e6e6']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor':  'black',
    'axes.linewidth':  0.8,
    'grid.color':      '#cccccc',
    'grid.linewidth':  0.6,
    'grid.linestyle':  '--',
})


# ── 1. CI Trends  (line chart) ─────────────────────────────────────────────────
def plot_ci_trends(results_dict, title="Consistency Index (CI) over 10 Rounds"):
    """Black-and-white line chart of CI values across rounds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (llm, values) in enumerate(results_dict.items()):
        ax.plot(
            range(1, len(values) + 1), values,
            marker    = MARKERS[i % len(MARKERS)],
            linestyle = LINESTYLES[i % len(LINESTYLES)],
            color     = GRAYS[i % len(GRAYS)],
            linewidth = 1.8,
            markersize = 6,
            markerfacecolor = 'white',
            markeredgecolor = GRAYS[i % len(GRAYS)],
            markeredgewidth = 1.2,
            label     = llm,
        )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Rounds", fontsize=11)
    ax.set_ylabel("CI Value", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, framealpha=1, edgecolor='black')
    ax.grid(True)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


# ── 2. Importance Counts  (grouped bar chart) ──────────────────────────────────
def plot_importance_counts(counts_data, title, criteria_names):
    """Black-and-white grouped bar chart with distinct hatch patterns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x         = np.arange(len(criteria_names))
    n_llms    = len(counts_data)
    width     = 0.7 / n_llms          # bars fill ~70% of each slot
    multiplier = 0

    for i, (llm, counts) in enumerate(counts_data.items()):
        offset = width * multiplier - (width * n_llms) / 2 + width / 2
        ax.bar(
            x + offset, counts, width,
            label       = llm,
            color       = GRAYS[i % len(GRAYS)],
            edgecolor   = 'black',
            linewidth   = 0.7,
            hatch       = HATCHES[i % len(HATCHES)],
        )
        multiplier += 1

    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_names, rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=9, framealpha=1, edgecolor='black')
    ax.grid(True, axis='y')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


# ── 3. Site Weights  (scatter / dot plot) ─────────────────────────────────────
# Full Alberta site names shown on x-axis (matches the reference figure)
SITE_LABELS = [
    "S1: Canmore",
    "S2: Lethbridge",
    "S3: Hanna",
    "S4: Fort McMurray",
    "S5: Medicine Hat",
    "S6: Cold Lake",
    "S7: Drumheller",
    "S8: Grand Prairie",
    "S9: Fort Chipewyan",
    "S10: High Level",
]

def plot_weights_for_sites(site_weights, criteria_name, llm_name):
    """
    Black-and-white scatter/dot plot for site weights across rounds.

    Font sizes are intentionally large — the caller scales this figure
    down ~50% for inclusion in a report, so everything must still be legible.
    """
    BASE = 22          # base font size (will read ~11 pt at 50 % scale)
    TICK = 20
    LEG  = 18
    MS   = 11          # marker size

    rounds = len(site_weights)
    sites  = SITE_LABELS[:len(site_weights[0])] if site_weights else SITE_LABELS

    fig, ax = plt.subplots(figsize=(14, 7))

    for r in range(rounds):
        ax.scatter(
            sites, site_weights[r],
            marker          = MARKERS[r % len(MARKERS)],
            s               = MS ** 2,
            color           = GRAYS[r % len(GRAYS)],
            edgecolors      = 'black',
            linewidths      = 0.8,
            label           = f"Round {r + 1}",
            zorder          = 3,
            alpha           = 0.85,
        )

    ax.set_title(
        f"Weights of {llm_name} under {criteria_name}",
        fontsize=BASE + 2, fontweight='bold', pad=14,
    )
    ax.set_ylabel("Weight Values", fontsize=BASE)
    ax.tick_params(axis='x', labelsize=TICK, rotation=45)
    ax.tick_params(axis='y', labelsize=TICK)

    # X-tick labels: rotate and align so they don't overlap at 50 % scale
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=45, ha='right', fontsize=TICK)

    # Legend — two columns to keep it compact
    ncol = 5 if rounds >= 6 else rounds
    leg = ax.legend(
        fontsize   = LEG,
        framealpha = 1,
        edgecolor  = 'black',
        ncol       = ncol,
        loc        = 'upper center',
        bbox_to_anchor = (0.5, 1.14),
    )

    ax.grid(True, alpha=0.4)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.93])   # leave room for top legend
    return fig