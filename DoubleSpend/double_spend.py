"""
Bitcoin Double-Spend Attack Analysis with GPU Acceleration
===========================================================
This script analyzes the probability of successful double-spend attacks on the Bitcoin network
using both analytical methods (Nakamoto's formula) and GPU-accelerated Monte Carlo simulations.
"""

import cupy as cp  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.stats import poisson, binom
import scipy.stats as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# for reproducibility
np.random.seed(42)
cp.random.seed(42)

#parameter settings for report generation
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# -------------------------------------------------------------------
# Analytical attack success probability (Nakamoto's formula)
# -------------------------------------------------------------------
def attack_success_probability(q, k):
    """
    Calculate the analytical probability that an attacker with hash power q
    succeeds in a double-spend attack after k confirmations.
    
    Based on Satoshi Nakamoto's analysis in the Bitcoin whitepaper (2008).
    
    Parameters:
    -----------
    q : float
        Fraction of total network hash power controlled by attacker (0 < q < 1)
    k : int
        Number of confirmations the merchant waits before accepting transaction
    
    Returns:
    --------
    float
        Probability of successful double-spend attack
    """
    if q >= 0.5:
        return 1.0  # Attacker with ≥50% hash power always wins eventually
    if k == 0:
        return 1.0  # Zero confirmations = guaranteed success
    
    p = 1 - q  # Honest miners' hash power
    λ = k * q / p  # Expected number of blocks attacker mines while honest chain grows by k
    
    # Calculate Poisson distribution for attacker's progress
    z_vals = np.arange(0, k+1)
    poisson_probs = poisson.pmf(z_vals, λ)
    
    # Nakamoto's formula: sum over all possible attacker progress values
    terms = poisson_probs * (1 - (q/p) ** (k - z_vals))
    prob = 1 - terms.sum()
    return max(0.0, prob)


# -------------------------------------------------------------------
# Monte Carlo Simulation with GPU (vectorized)
# -------------------------------------------------------------------
def simulate_attack_gpu(q, k, N=1_000_000, verbose=False):
    """
    GPU-accelerated Monte Carlo simulation of double-spend attack attempts.
    
    Simulates N independent attack scenarios where:
    1. Honest chain has k-block head start
    2. Attacker tries to catch up by mining a longer chain
    3. Each block is mined by attacker with probability q, honest with probability (1-q)
    
    Parameters:
    -----------
    q : float
        Attacker's hash power fraction
    k : int
        Number of confirmations (honest chain head start)
    N : int
        Number of Monte Carlo trials
    verbose : bool
        Print progress information
    
    Returns:
    --------
    tuple (probability, confidence_interval)
        - probability: Estimated attack success rate
        - confidence_interval: 95% binomial confidence interval
    """
    if verbose:
        print(f"  Simulating {N:,} trials for q={q:.2f}, k={k}...")
    
    # Maximum simulation steps (prevent infinite loops)
    max_steps = k + 100
    
    # Generate all random block outcomes on GPU
    rand = cp.random.rand(N, max_steps)
    
    # Track cumulative blocks mined by each party
    # Attacker starts at 0, honest chain starts at k
    attacker_blocks = (rand < q).astype(cp.int32).cumsum(axis=1)
    honest_blocks = (rand >= q).astype(cp.int32).cumsum(axis=1) + k
    
    # Attack succeeds if attacker catches up at any point
    success = (attacker_blocks >= honest_blocks).any(axis=1)
    successes = int(success.sum().get())  # Transfer from GPU to CPU
    
    prob = successes / N
    
    # Calculate 95% binomial confidence interval (Wilson score interval)
    if successes == 0:
        ci_low, ci_high = 0, 3.84 / (N + 3.84)
    elif successes == N:
        ci_low, ci_high = N / (N + 3.84), 1
    else:
        ci_low, ci_high = st.binom.interval(0.95, N, prob, loc=0)
        ci_low, ci_high = ci_low / N, ci_high / N
    
    return prob, (ci_low, ci_high)


# -------------------------------------------------------------------
# Economic analysis
# -------------------------------------------------------------------
def calculate_economic_metrics(q, k, block_reward=6.25, tx_value=1000):
    """
    Calculate economic metrics for double-spend attacks.
    
    Parameters:
    -----------
    q : float
        Attacker's hash power
    k : int
        Number of confirmations
    block_reward : float
        BTC reward per block (currently 6.25 BTC)
    tx_value : float
        Value of transaction being double-spent (in BTC)
    
    Returns:
    --------
    dict
        Economic metrics including expected cost and profit
    """
    p_success = attack_success_probability(q, k)
    
    # Expected blocks to mine = k / q (on average)
    expected_blocks = k / q if q > 0 else float('inf')
    
    # Cost = opportunity cost of mining on attack chain instead of honest chain
    attack_cost = expected_blocks * block_reward * q
    
    # Expected profit
    expected_profit = p_success * tx_value - attack_cost
    
    # Break-even transaction value
    breakeven_value = attack_cost / p_success if p_success > 0 else float('inf')
    
    return {
        'success_probability': p_success,
        'expected_blocks': expected_blocks,
        'attack_cost': attack_cost,
        'expected_profit': expected_profit,
        'breakeven_value': breakeven_value,
        'roi': (expected_profit / attack_cost * 100) if attack_cost > 0 else 0
    }


# -------------------------------------------------------------------
# Main Analysis
# -------------------------------------------------------------------
def main():
    print("="*80)
    print("BITCOIN DOUBLE-SPEND ATTACK ANALYSIS")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parameters
    qs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]  # More granular
    k_max = 20
    N = 1_000_000  # Number of Monte Carlo trials
    
    # Standard confirmation thresholds
    standard_confirmations = {
        1: "High-frequency trading",
        3: "Standard e-commerce", 
        6: "Bitcoin standard (exchanges)",
        10: "High-value transactions",
        100: "Coinbase maturity"
    }
    
    # Storage for results
    analytical_prob = {q: [] for q in qs}
    simulated_prob = {q: [] for q in qs}
    simulated_ci = {q: [] for q in qs}
    
    print("Computing attack success probabilities...\n")
    
    # Compute probabilities
    for q in qs:
        print(f"Processing q = {q:.2f} ({q*100:.0f}% attacker hash power)")
        for k in range(k_max + 1):
            # Analytical calculation
            analytical_prob[q].append(attack_success_probability(q, k))
            
            # GPU simulation for smaller k values (computationally intensive for large k)
            if k <= 10:
                prob, ci = simulate_attack_gpu(q, k, N, verbose=False)
                simulated_prob[q].append(prob)
                simulated_ci[q].append(ci)
            else:
                simulated_prob[q].append(None)
                simulated_ci[q].append((None, None))
    
    # Save detailed results to CSV
    print("\nSaving results...")
    rows = []
    for q in qs:
        for k in range(k_max + 1):
            row = {
                "attacker_hashpower": q,
                "confirmations": k,
                "analytical_probability": analytical_prob[q][k],
                "simulated_probability": simulated_prob[q][k],
                "ci_low": simulated_ci[q][k][0],
                "ci_high": simulated_ci[q][k][1],
            }
            
            # Add economic metrics for selected confirmation counts
            if k in [1, 3, 6, 10]:
                metrics = calculate_economic_metrics(q, k)
                row.update({
                    "attack_cost_btc": metrics['attack_cost'],
                    "breakeven_tx_value_btc": metrics['breakeven_value']
                })
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("attack_success_probabilities_detailed.csv", index=False)
    print("Results saved to attack_success_probabilities_detailed.csv")
    
    # -------------------------------------------------------------------
    # Enhanced Visualization
    # -------------------------------------------------------------------
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(18, 14))  # Increased from (16, 12)
    fig.suptitle("Bitcoin Double-Spend Attack Analysis: Comprehensive Study", 
                fontsize=16, fontweight='bold', y=0.995)  # Adjusted from y=1.02

    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.45)  # Increased spacing
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(qs)))
    
    # ---- Subplot 1: Main analytical curves ----
    ax1 = fig.add_subplot(gs[0, :2])
    for i, q in enumerate(qs):
        if q in [0.1, 0.2, 0.3, 0.4]:  # Highlight key values
            ax1.semilogy(range(k_max + 1), analytical_prob[q], 
                        label=f"{int(q*100)}% hash power", 
                        color=colors[i], linewidth=2.5, marker='o', markersize=4)
        else:
            ax1.semilogy(range(k_max + 1), analytical_prob[q], 
                        color=colors[i], linewidth=1, alpha=0.5)
    
    # Add confirmation thresholds
    for k_thresh, desc in standard_confirmations.items():
        if k_thresh <= k_max:
            ax1.axvline(k_thresh, color="gray", linestyle="--", alpha=0.4)
            ax1.text(k_thresh + 0.1, 1e-8, desc, rotation=90, fontsize=8, alpha=0.7)
    
    ax1.set_xlabel("Number of Confirmations (k)")
    ax1.set_ylabel("Attack Success Probability (log scale)")
    ax1.set_title("Nakamoto's Formula: Attack Success vs. Confirmations", fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim([1e-10, 1])
    ax1.set_xlim([0, k_max])
    
    # Add annotations for key insights
    ax1.annotate('6 confirmations:\nBitcoin standard', 
                xy=(6, analytical_prob[0.1][6]), 
                xytext=(8, 1e-4),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    # ---- Subplot 2: Simulation validation ----
    ax2 = fig.add_subplot(gs[0, 2])
    q_compare = 0.25  # Compare for 25% attacker
    k_vals = [k for k in range(11) if simulated_prob[q_compare][k] is not None]
    analytical_vals = [analytical_prob[q_compare][k] for k in k_vals]
    simulated_vals = [simulated_prob[q_compare][k] for k in k_vals]
    ci_vals = [simulated_ci[q_compare][k] for k in k_vals]
    
    # Plot with error bars
    yerr = [[s - ci[0] for s, ci in zip(simulated_vals, ci_vals)],
            [ci[1] - s for s, ci in zip(simulated_vals, ci_vals)]]
    
    ax2.plot(k_vals, analytical_vals, 'b-', linewidth=2, label='Analytical')
    ax2.errorbar(k_vals, simulated_vals, yerr=yerr, fmt='ro', 
                capsize=3, markersize=6, label=f'Simulated (N={N:,})')
    ax2.set_xlabel("Confirmations (k)")
    ax2.set_ylabel("Success Probability")
    ax2.set_title(f"Model Validation\n(q={q_compare:.0%} attacker)", fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_ylim([1e-6, 1])
    
    # ---- Subplot 3: Heatmap of success probabilities ----
    ax3 = fig.add_subplot(gs[1, :2])
    k_range = range(0, 11)
    q_range = qs[:7]  # Limit for readability
    
    heatmap_data = np.array([[analytical_prob[q][k] for k in k_range] for q in q_range])
    im = ax3.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', 
                   norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=1))
    
    ax3.set_xticks(range(len(k_range)))
    ax3.set_xticklabels(k_range)
    ax3.set_yticks(range(len(q_range)))
    ax3.set_yticklabels([f"{int(q*100)}%" for q in q_range])
    ax3.set_xlabel("Number of Confirmations")
    ax3.set_ylabel("Attacker Hash Power")
    ax3.set_title("Attack Success Probability Heatmap", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Success Probability', rotation=270, labelpad=15)
    
    # Add text annotations for key values
    for i, q in enumerate(q_range):
        for j, k in enumerate(k_range):
            prob = heatmap_data[i, j]
            if prob > 0.01:  # Only show significant probabilities
                text = ax3.text(j, i, f'{prob:.2f}' if prob > 0.1 else f'{prob:.3f}',
                              ha="center", va="center", color="white" if prob > 0.5 else "black",
                              fontsize=7)
    
    # ---- Subplot 4: Economic analysis ----
    ax4 = fig.add_subplot(gs[1, 2])
    k_economic = 6  # Standard 6 confirmations
    economic_data = []
    
    for q in [0.1, 0.2, 0.3, 0.4]:
        metrics = calculate_economic_metrics(q, k_economic)
        economic_data.append(metrics['breakeven_value'])
    
    bars = ax4.bar([f"{int(q*100)}%" for q in [0.1, 0.2, 0.3, 0.4]], 
                   economic_data, color=['green', 'yellow', 'orange', 'red'])
    ax4.set_ylabel("Break-even Transaction Value (BTC)")
    ax4.set_title(f"Economic Threshold\n({k_economic} confirmations)", fontweight='bold')
    ax4.set_yscale('log')
    
    # Add value labels on bars
    for bar, value in zip(bars, economic_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{value:.1f} BTC', ha='center', va='bottom', fontsize=8)
    
    # ---- Subplot 5: Time analysis ----
    ax5 = fig.add_subplot(gs[2, 0])
    confirm_times = [0, 10, 30, 60, 100, 200]  # minutes
    confirm_blocks = [t // 10 for t in confirm_times]  # ~10 min per block
    
    for q in [0.1, 0.25, 0.4]:
        probs = [attack_success_probability(q, k) for k in confirm_blocks]
        ax5.semilogy(confirm_times, probs, marker='o', 
                    label=f"{int(q*100)}% attacker", linewidth=2)
    
    ax5.set_xlabel("Waiting Time (minutes)")
    ax5.set_ylabel("Attack Success Probability")
    ax5.set_title("Security vs. Waiting Time", fontweight='bold')
    ax5.legend()
    ax5.grid(True, which="both", alpha=0.3)
    
    # ---- Subplot 6: Comparison table ----
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create comparison table
    table_data = []
    table_data.append(['Confirmations', 'Time (min)', '10% Attacker', '25% Attacker', '40% Attacker'])
    
    for k in [0, 1, 3, 6, 10]:
        row = [str(k), f"{k*10}", 
               f"{analytical_prob[0.1][k]:.2e}", 
               f"{analytical_prob[0.25][k]:.2e}",
               f"{analytical_prob[0.4][k]:.2e}"]
        table_data.append(row)
    
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style the header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the probability cells
    for i in range(1, 6):
        for j in range(2, 5):
            val = float(table_data[i][j])
            if val < 1e-6:
                color = '#90EE90'  # Light green
            elif val < 1e-3:
                color = '#FFFFE0'  # Light yellow
            elif val < 0.1:
                color = '#FFD700'  # Gold
            else:
                color = '#FF6B6B'  # Light red
            table[(i, j)].set_facecolor(color)
    
    ax6.set_title("Attack Success Probability Summary Table", fontweight='bold', pad=20)
    
    # Add footnote
    fig.text(0.5, 0.02, 
            "Note: Analysis based on Nakamoto's model assuming constant hash rate, no network latency, and rational attacker behavior.\n" +
            f"Monte Carlo simulations performed with {N:,} trials on GPU. Current block reward: 6.25 BTC.",
            ha='center', fontsize=9, style='italic', wrap=True)
    
    # Save figures
    plt.savefig("double_spend_analysis_enhanced.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig("double_spend_analysis_enhanced.png", dpi=300, bbox_inches="tight")
    print("Visualizations saved as PDF and PNG")
    
    # -------------------------------------------------------------------
    # Print summary statistics
    # -------------------------------------------------------------------
    print("\n" + "="*80)
    print("ATTACK SUCCESS PROBABILITIES (Selected Values)")
    print("="*80)
    
    for k in [0, 1, 3, 6, 10]:
        print(f"\n{'='*50}")
        print(f"CONFIRMATIONS: {k} (~{k*10} minutes wait time)")
        print(f"{'='*50}")
        
        for q in [0.05, 0.1, 0.25, 0.4]:
            prob = analytical_prob[q][k]
            metrics = calculate_economic_metrics(q, k)
            
            print(f"\nAttacker with {q*100:>2.0f}% hash power:")
            print(f"  • Success probability: {prob:.6e} ({prob*100:.4f}%)")
            print(f"  • Attack cost: {metrics['attack_cost']:.2f} BTC")
            print(f"  • Break-even TX value: {metrics['breakeven_value']:.2f} BTC")
            
            # Risk assessment
            if prob < 1e-6:
                risk = "NEGLIGIBLE"
            elif prob < 1e-3:
                risk = "VERY LOW"
            elif prob < 0.01:
                risk = "LOW"
            elif prob < 0.1:
                risk = "MODERATE"
            else:
                risk = "HIGH"
            print(f"  • Risk level: {risk}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    main()
