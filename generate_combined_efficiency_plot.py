#!/usr/bin/env python3
"""
Generate Combined Efficiency Loss comparison plot
Shows actual Network Efficiency values (not percentage change)
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load resilience data
strategies = {
    'Degree Removal': 'degree_removal/results.json',
    'Betweenness Removal': 'betweenness_removal/results.json',
    'Closeness Removal': 'closeness_removal/results.json',
    'Route Removal': 'route_removal/results.json'
}

colors = {
    'Degree Removal': '#3498db',       # Blue
    'Betweenness Removal': '#e74c3c',  # Red
    'Closeness Removal': '#9b59b6',    # Purple
    'Route Removal': '#f39c12'         # Orange
}

# Create figure
plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8-darkgrid')

# Plot each strategy
for strategy_name, json_file in strategies.items():
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get removal analysis data
        removal_analysis = data.get('removal_analysis', {})
        
        # Extract efficiency data (key is 'network_efficiency')
        efficiency_data = removal_analysis.get('network_efficiency', [])
        
        if not efficiency_data:
            print(f"✗ Warning: No efficiency data in {json_file}")
            continue
        
        # Create removals sequence (0, 1, 2, 3, ... 15)
        removals = list(range(len(efficiency_data)))
        
        # Plot
        plt.plot(removals, efficiency_data, 
                marker='o', 
                linewidth=2.5, 
                markersize=6,
                color=colors[strategy_name],
                label=strategy_name,
                alpha=0.9)
        
        print(f"✓ Loaded {strategy_name}: {len(removals)} data points (0 to {max(removals)} removals)")
        print(f"   Efficiency range: {min(efficiency_data):.4f} to {max(efficiency_data):.4f}")
        
    except FileNotFoundError:
        print(f"✗ Warning: {json_file} not found")
    except Exception as e:
        print(f"✗ Error loading {strategy_name}: {e}")

# Customize plot
plt.xlabel('Number of Nodes/Routes Removed', fontsize=14, fontweight='bold')
plt.ylabel('Network Efficiency', fontsize=14, fontweight='bold')
plt.title('Combined Efficiency Loss: All Removal Strategies', 
          fontsize=16, fontweight='bold', pad=20)

plt.legend(loc='upper right', fontsize=12, framealpha=0.95, 
          shadow=True, borderpad=1)

plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Set y-axis limits
plt.ylim(0.0, 0.17)

# Add background color
ax = plt.gca()
ax.set_facecolor('#f8f9fa')

# Save
output_file = 'combined_efficiency_loss.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output_file}")

plt.close()
