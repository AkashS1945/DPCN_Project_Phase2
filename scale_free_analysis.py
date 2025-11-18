#!/usr/bin/env python3
"""
Scale-Free Network Analysis
Determines if the network follows a power-law degree distribution.
"""

import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json


def power_law(x, gamma, c):
    """Power law function: P(k) = c * k^(-gamma)"""
    return c * np.power(x, -gamma)


def analyze_scale_free(nodes_file='nodes.csv', edges_file='edges.csv', nodes_df=None, edges_df=None):
    """
    Comprehensive scale-free analysis.
    
    Args:
        nodes_file: Path to nodes CSV (used if nodes_df is None)
        edges_file: Path to edges CSV (used if edges_df is None)
        nodes_df: Pandas dataframe of nodes (optional)
        edges_df: Pandas dataframe of edges (optional)
    
    Returns:
        dict: Analysis results including gamma, R², KS statistic, and verdict
    """
    print("🔬 Scale-Free Network Analysis")
    print("=" * 60)
    
    # Load data (use dataframes if provided, otherwise read from files)
    if nodes_df is None:
        nodes = pd.read_csv(nodes_file)
    else:
        nodes = nodes_df
    
    if edges_df is None:
        edges = pd.read_csv(edges_file)
    else:
        edges = edges_df

    
    # Build degree sequence
    print("\n📊 Step 1: Computing node degrees...")
    degree_sequence = []
    for node_id in nodes['node_id']:
        degree = len(edges[(edges['from_id'] == node_id) | (edges['to_id'] == node_id)])
        degree_sequence.append(degree)
    
    print(f"   Total nodes: {len(degree_sequence)}")
    print(f"   Degree range: {min(degree_sequence)} to {max(degree_sequence)}")
    print(f"   Average degree: {np.mean(degree_sequence):.2f}")
    
    # Create degree distribution
    print("\n📈 Step 2: Creating degree distribution...")
    degree_counts = Counter(degree_sequence)
    degrees = np.array(sorted(degree_counts.keys()))
    counts = np.array([degree_counts[d] for d in degrees])
    probabilities = counts / len(degree_sequence)
    
    # Filter out zero degree nodes and create log-log data
    mask = (degrees > 0) & (counts > 0)
    degrees_nonzero = degrees[mask]
    probs_nonzero = probabilities[mask]
    
    print(f"   Unique degree values: {len(degrees)}")
    print(f"   Non-zero degrees for fitting: {len(degrees_nonzero)}")
    
    # Fit power law using curve_fit
    print("\n🔧 Step 3: Fitting power law P(k) = c * k^(-gamma)...")
    try:
        # Initial guess for parameters
        popt, pcov = curve_fit(power_law, degrees_nonzero, probs_nonzero, 
                              p0=[2.5, 0.1], maxfev=10000)
        gamma, c = popt
        
        # Calculate fitted values
        fitted_probs = power_law(degrees_nonzero, gamma, c)
        
        # Calculate R² (coefficient of determination)
        ss_res = np.sum((probs_nonzero - fitted_probs) ** 2)
        ss_tot = np.sum((probs_nonzero - np.mean(probs_nonzero)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((probs_nonzero - fitted_probs) ** 2))
        
        print(f"   ✓ Power-law exponent (gamma): {gamma:.4f}")
        print(f"   ✓ Coefficient (c): {c:.6f}")
        print(f"   ✓ R² score: {r_squared:.4f}")
        print(f"   ✓ RMSE: {rmse:.6f}")
        
    except Exception as e:
        print(f"   ✗ Curve fitting failed: {e}")
        gamma, c, r_squared, rmse = 0, 0, 0, 1
        fitted_probs = np.zeros_like(probs_nonzero)
    
    # Kolmogorov-Smirnov test
    print("\n📐 Step 4: Kolmogorov-Smirnov goodness-of-fit test...")
    
    # Create empirical CDF
    sorted_degrees = np.sort(degree_sequence)
    empirical_cdf = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
    
    # Create theoretical power-law CDF (simplified)
    theoretical_cdf = np.array([np.sum(power_law(np.arange(1, d+1), gamma, c)) 
                                for d in sorted_degrees])
    # Normalize
    if np.max(theoretical_cdf) > 0:
        theoretical_cdf = theoretical_cdf / np.max(theoretical_cdf)
    
    # KS statistic (maximum distance between CDFs)
    ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
    
    print(f"   ✓ KS statistic: {ks_statistic:.4f}")
    print(f"     (Lower is better; < 0.1 is good)")
    
    # Linear regression in log-log space (alternative method)
    print("\n📏 Step 5: Linear regression in log-log space...")
    log_degrees = np.log10(degrees_nonzero)
    log_probs = np.log10(probs_nonzero)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_probs)
    gamma_loglog = -slope  # Negative because P(k) ~ k^(-gamma)
    r_squared_loglog = r_value ** 2
    
    print(f"   ✓ Gamma (from log-log): {gamma_loglog:.4f}")
    print(f"   ✓ R² (log-log): {r_squared_loglog:.4f}")
    print(f"   ✓ P-value: {p_value:.6f}")
    
    # Decision criteria
    print("\n🎯 Step 6: Scale-Free Assessment...")
    print("-" * 60)
    
    # Criteria for scale-free network
    criteria = []
    
    # Criterion 1: Gamma in typical range [2, 3]
    gamma_in_range = 2.0 <= gamma <= 3.0
    gamma_loglog_in_range = 2.0 <= gamma_loglog <= 3.0
    criteria.append(("Gamma ∈ [2, 3]", gamma_in_range or gamma_loglog_in_range))
    
    # Criterion 2: Good fit quality
    good_fit = (r_squared > 0.5 or r_squared_loglog > 0.5) and rmse < 0.1
    criteria.append(("Good fit (R² > 0.5, RMSE < 0.1)", good_fit))
    
    # Criterion 3: KS test
    ks_good = ks_statistic < 0.15
    criteria.append(("KS statistic < 0.15", ks_good))
    
    # Print criteria results
    for criterion, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {criterion}")
    
    # Final verdict
    passed_count = sum(1 for _, passed in criteria if passed)
    is_scale_free = passed_count >= 2  # At least 2 out of 3 criteria
    
    print("-" * 60)
    if is_scale_free:
        print("✅ VERDICT: This network is LIKELY SCALE-FREE")
        print(f"   The degree distribution follows a power law with γ ≈ {gamma:.2f}")
        print(f"   {passed_count}/3 criteria passed")
    else:
        print("❌ VERDICT: This network is NOT scale-free")
        print(f"   The degree distribution does NOT follow a power law")
        print(f"   Only {passed_count}/3 criteria passed")
    
    # Additional insights
    print("\n💡 Network Insights:")
    if gamma < 2:
        print("   • Very small γ suggests super-connected hubs (highly centralized)")
    elif 2 <= gamma <= 3:
        print("   • γ ∈ [2,3] is typical for real-world scale-free networks")
        print("   • Few highly-connected hubs, many low-degree nodes")
    else:
        print("   • Large γ suggests more homogeneous degree distribution")
    
    # Create results dictionary
    results = {
        'is_scale_free': is_scale_free,
        'gamma': round(float(gamma), 4),
        'gamma_loglog': round(float(gamma_loglog), 4),
        'coefficient_c': round(float(c), 6),
        'r_squared': round(float(r_squared), 4),
        'r_squared_loglog': round(float(r_squared_loglog), 4),
        'rmse': round(float(rmse), 6),
        'ks_statistic': round(float(ks_statistic), 4),
        'p_value': round(float(p_value), 6),
        'criteria_passed': passed_count,
        'criteria_total': len(criteria),
        'degree_distribution': [
            {'degree': int(d), 'count': int(c), 'probability': round(float(p), 6)}
            for d, c, p in zip(degrees, counts, probabilities)
        ],
        'network_stats': {
            'total_nodes': len(degree_sequence),
            'min_degree': int(min(degree_sequence)),
            'max_degree': int(max(degree_sequence)),
            'avg_degree': round(float(np.mean(degree_sequence)), 2),
            'median_degree': int(np.median(degree_sequence))
        }
    }
    
    # Save results
    print("\n💾 Saving results to scale_free_results.json...")
    with open('scale_free_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    print("📊 Creating visualization...")
    create_visualization(degrees_nonzero, probs_nonzero, fitted_probs, gamma, r_squared)
    
    print("\n✅ Analysis complete!")
    print("=" * 60)
    
    return results


def create_visualization(degrees, empirical_probs, fitted_probs, gamma, r_squared):
    """Create log-log plot of degree distribution with power-law fit."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Log-log scale
    ax1.scatter(degrees, empirical_probs, alpha=0.6, s=50, label='Empirical', color='blue')
    ax1.plot(degrees, fitted_probs, 'r-', linewidth=2, label=f'Power Law Fit (γ={gamma:.2f})')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degree (k)', fontsize=12)
    ax1.set_ylabel('P(k)', fontsize=12)
    ax1.set_title(f'Degree Distribution (Log-Log Scale)\nR² = {r_squared:.4f}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regular scale
    ax2.scatter(degrees, empirical_probs, alpha=0.6, s=50, label='Empirical', color='blue')
    ax2.plot(degrees, fitted_probs, 'r-', linewidth=2, label=f'Power Law Fit')
    ax2.set_xlabel('Degree (k)', fontsize=12)
    ax2.set_ylabel('P(k)', fontsize=12)
    ax2.set_title('Degree Distribution (Linear Scale)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scale_free_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved visualization to scale_free_analysis.png")
    plt.close()


if __name__ == '__main__':
    results = analyze_scale_free()
