#!/usr/bin/env python3
"""
Compare Network Efficiency Across All 4 Removal Analyses
Shows which removal strategy has the biggest impact on network resilience
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Load results
analyses = {
    'Degree Removal': 'degree_removal/results.json',
    'Betweenness Removal': 'betweenness_removal/results.json',
    'Closeness Removal': 'closeness_removal/results.json',
    'Route Removal': 'route_removal/results.json'
}

colors = {
    'Degree Removal': '#e74c3c',
    'Betweenness Removal': '#3498db',
    'Closeness Removal': '#2ecc71',
    'Route Removal': '#f39c12'
}

# Load data
data = {}
for name, path in analyses.items():
    with open(path, 'r') as f:
        data[name] = json.load(f)

# Create comparison plots
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('Network Resilience Comparison Across 4 Removal Strategies', 
             fontsize=18, fontweight='bold', y=0.995)

metrics = [
    ('avg_clustering', 'Average Clustering Coefficient'),
    ('avg_betweenness', 'Average Betweenness Centrality'),
    ('avg_closeness', 'Average Closeness Centrality'),
    ('avg_degree', 'Average Node Degree'),
    ('avg_time', 'Average Travel Time (min)'),
    ('avg_cost', 'Average Travel Cost (₹)'),
    ('network_efficiency', 'Network Efficiency'),
    ('num_components', 'Number of Connected Components'),
    ('largest_component_size', 'Largest Component Size')
]

for idx, (metric_key, metric_name) in enumerate(metrics):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Plot each analysis
    for analysis_name, results in data.items():
        removal_data = results['removal_analysis']
        ax.plot(removal_data['removal_sequence'], 
               removal_data[metric_key],
               marker='o', linewidth=2.5, markersize=7,
               color=colors[analysis_name],
               label=analysis_name,
               alpha=0.8)
    
    ax.set_xlabel('Removal Sequence (1st → 15th)', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('comparison_all_analyses.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_all_analyses.png")

# Create efficiency comparison only (larger)
fig, ax = plt.subplots(figsize=(14, 8))

for analysis_name, results in data.items():
    removal_data = results['removal_analysis']
    efficiency = removal_data['network_efficiency']
    baseline = efficiency[0]
    
    # Calculate percentage change
    pct_change = [(val - baseline) / baseline * 100 for val in efficiency]
    
    ax.plot(removal_data['removal_sequence'], 
           pct_change,
           marker='o', linewidth=3, markersize=10,
           color=colors[analysis_name],
           label=analysis_name,
           alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax.set_xlabel('Removal Sequence (1st → 15th)', fontsize=13, fontweight='bold')
ax.set_ylabel('Network Efficiency Change (%)', fontsize=13, fontweight='bold')
ax.set_title('Network Efficiency Degradation: 4 Removal Strategies Compared',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')

# Add text annotations for final values
for analysis_name, results in data.items():
    removal_data = results['removal_analysis']
    efficiency = removal_data['network_efficiency']
    baseline = efficiency[0]
    final_pct = (efficiency[-1] - baseline) / baseline * 100
    
    ax.annotate(f'{final_pct:.1f}%',
               xy=(15, final_pct),
               xytext=(15.5, final_pct),
               fontsize=11,
               fontweight='bold',
               color=colors[analysis_name])

plt.tight_layout()
plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: efficiency_comparison.png")

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for analysis_name, results in data.items():
    print(f"\n{analysis_name}:")
    removal_data = results['removal_analysis']
    
    # Calculate changes
    eff_baseline = removal_data['network_efficiency'][0]
    eff_final = removal_data['network_efficiency'][-1]
    eff_change = (eff_final - eff_baseline) / eff_baseline * 100
    
    deg_baseline = removal_data['avg_degree'][0]
    deg_final = removal_data['avg_degree'][-1]
    deg_change = (deg_final - deg_baseline) / deg_baseline * 100
    
    print(f"  Network Efficiency: {eff_baseline:.4f} → {eff_final:.4f} ({eff_change:+.2f}%)")
    print(f"  Average Degree: {deg_baseline:.2f} → {deg_final:.2f} ({deg_change:+.2f}%)")
    print(f"  Components: {removal_data['num_components'][0]:.0f} → {removal_data['num_components'][-1]:.0f}")

# Identify most impactful analysis
print("\n" + "=" * 80)
print("MOST IMPACTFUL REMOVAL STRATEGY")
print("=" * 80)

impacts = {}
for analysis_name, results in data.items():
    removal_data = results['removal_analysis']
    eff_baseline = removal_data['network_efficiency'][0]
    eff_final = removal_data['network_efficiency'][-1]
    impacts[analysis_name] = abs((eff_final - eff_baseline) / eff_baseline * 100)

most_impactful = max(impacts, key=impacts.get)
print(f"\n🔴 {most_impactful}: {impacts[most_impactful]:.2f}% efficiency loss")
print(f"\nThis indicates that {most_impactful.lower()} targets the most")
print(f"critical nodes/routes for network performance!")

print("\n" + "=" * 80)
print("✅ Comparison plots generated!")
print("=" * 80)
print("\nGenerated files:")
print("  • comparison_all_analyses.png - All 9 metrics compared")
print("  • efficiency_comparison.png - Network efficiency comparison")
print("=" * 80)
