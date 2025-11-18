#!/usr/bin/env python3
"""
Interactive Data Explorer for Resilience Analysis Results
View and analyze the resilience analysis data
"""

import json
import pandas as pd
from pathlib import Path


def print_section(title, char="="):
    """Print a formatted section header"""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def load_results(filename="resilience_analysis_results.json"):
    """Load the analysis results"""
    with open(filename, 'r') as f:
        return json.load(f)


def display_top_hubs(data, n=15):
    """Display top N hubs in a formatted table"""
    print_section("TOP CRITICAL HUBS", "=")
    
    hubs = data['top_hubs'][:n]
    
    # Create DataFrame for nice formatting
    df_data = []
    for hub in hubs:
        df_data.append({
            'Rank': hub['rank'],
            'Node ID': hub['node_id'],
            'Name': hub['name'][:25],  # Truncate long names
            'Layer': hub['layer'],
            'Score': f"{hub['composite_score']:.4f}",
            'Degree': hub['degree'],
            'Betweenness': f"{hub['betweenness_centrality']:.4f}",
            'Closeness': f"{hub['closeness_centrality']:.4f}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Layer distribution
    print("\n" + "-" * 80)
    print("Hub Distribution by Layer:")
    layer_counts = {}
    for hub in hubs:
        layer = hub['layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
        print(f"  {layer.upper():<10}: {count:>2} hubs ({count/len(hubs)*100:>5.1f}%)")


def display_top_routes(data, n=15):
    """Display top N routes in a formatted table"""
    print_section("TOP CRITICAL ROUTES", "=")
    
    routes = data['top_routes'][:n]
    
    # Create DataFrame for nice formatting
    df_data = []
    for route in routes:
        df_data.append({
            'Rank': route['rank'],
            'From': route['from_name'][:20],  # Truncate
            'To': route['to_name'][:20],  # Truncate
            'Mode': route['mode'],
            'Score': f"{route['score']:.4f}",
            'Distance': f"{route['distance_km']:.2f} km",
            'Time': f"{route['time_min']:.1f} min",
            'Cost': f"₹{route['cost_rs']:.0f}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Mode distribution
    print("\n" + "-" * 80)
    print("Route Distribution by Mode:")
    mode_counts = {}
    for route in routes:
        mode = route['mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode.upper():<10}: {count:>2} routes ({count/len(routes)*100:>5.1f}%)")


def display_hub_removal_impact(data):
    """Display the impact of hub removals"""
    print_section("HUB REMOVAL IMPACT ANALYSIS", "=")
    
    hub_analysis = data['hub_removal_analysis']
    
    print("Network Degradation Over Sequential Hub Removals:\n")
    
    # Key metrics to display
    metrics = [
        ('network_efficiency', 'Network Efficiency'),
        ('avg_clustering', 'Avg Clustering'),
        ('avg_betweenness', 'Avg Betweenness'),
        ('avg_closeness', 'Avg Closeness'),
        ('avg_degree', 'Avg Degree'),
        ('num_components', 'Components'),
    ]
    
    # Show initial, midpoint, and final values
    positions = [0, 7, 14]  # 1st, 8th, 15th removal
    labels = ['Initial (1st)', 'Midpoint (8th)', 'Final (15th)']
    
    print(f"{'Metric':<25}", end='')
    for label in labels:
        print(f"{label:>20}", end='')
    print(f"{'Change':>15}")
    print("-" * 95)
    
    for metric_key, metric_name in metrics:
        values = hub_analysis[metric_key]
        print(f"{metric_name:<25}", end='')
        
        for pos in positions:
            print(f"{values[pos]:>20.4f}", end='')
        
        change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        print(f"{change:>+14.2f}%")
    
    print("\n" + "-" * 80)
    print("\nMost Critical Hub Removals (Largest Impact on Efficiency):")
    
    # Calculate impact of each removal
    efficiency = hub_analysis['network_efficiency']
    impacts = []
    for i in range(len(efficiency) - 1):
        impact = efficiency[i] - efficiency[i+1] if i == 0 else efficiency[i-1] - efficiency[i]
        hub_details = hub_analysis['hub_details'][i]
        impacts.append((hub_details['name'], hub_details['layer'], impact))
    
    # Sort by impact
    impacts.sort(key=lambda x: -x[2])
    
    for i, (name, layer, impact) in enumerate(impacts[:5], 1):
        print(f"  {i}. {name:<30} ({layer:<6}) - {impact:.6f} efficiency loss")


def display_route_removal_impact(data):
    """Display the impact of route removals"""
    print_section("ROUTE REMOVAL IMPACT ANALYSIS", "=")
    
    route_analysis = data['route_removal_analysis']
    
    print("Network Degradation Over Sequential Route Removals:\n")
    
    # Key metrics to display
    metrics = [
        ('network_efficiency', 'Network Efficiency'),
        ('avg_clustering', 'Avg Clustering'),
        ('avg_betweenness', 'Avg Betweenness'),
        ('avg_closeness', 'Avg Closeness'),
        ('avg_degree', 'Avg Degree'),
        ('num_components', 'Components'),
    ]
    
    # Show initial, midpoint, and final values
    positions = [0, 7, 14]  # 1st, 8th, 15th removal
    labels = ['Initial (1st)', 'Midpoint (8th)', 'Final (15th)']
    
    print(f"{'Metric':<25}", end='')
    for label in labels:
        print(f"{label:>20}", end='')
    print(f"{'Change':>15}")
    print("-" * 95)
    
    for metric_key, metric_name in metrics:
        values = route_analysis[metric_key]
        print(f"{metric_name:<25}", end='')
        
        for pos in positions:
            print(f"{values[pos]:>20.4f}", end='')
        
        change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        print(f"{change:>+14.2f}%")
    
    print("\n" + "-" * 80)
    print("\nKey Observation:")
    
    eff_change = route_analysis['network_efficiency'][-1] - route_analysis['network_efficiency'][0]
    if abs(eff_change) < 0.001:
        print("  ✓ Network shows HIGH RESILIENCE to route removals")
        print("    Minimal efficiency degradation indicates strong redundancy")
    else:
        print(f"  ⚠ Network efficiency changed by {eff_change:.4f}")


def display_comparison(data):
    """Compare hub vs route removal impacts"""
    print_section("HUB vs ROUTE REMOVAL COMPARISON", "=")
    
    hub_analysis = data['hub_removal_analysis']
    route_analysis = data['route_removal_analysis']
    
    print("Comparing Impact on Key Metrics:\n")
    
    metrics = [
        ('network_efficiency', 'Network Efficiency'),
        ('avg_clustering', 'Clustering Coefficient'),
        ('avg_betweenness', 'Betweenness Centrality'),
        ('num_components', 'Connected Components'),
    ]
    
    print(f"{'Metric':<30}{'Hub Impact':>20}{'Route Impact':>20}{'More Critical':>20}")
    print("-" * 90)
    
    for metric_key, metric_name in metrics:
        hub_vals = hub_analysis[metric_key]
        route_vals = route_analysis[metric_key]
        
        hub_change = ((hub_vals[-1] - hub_vals[0]) / hub_vals[0] * 100) if hub_vals[0] != 0 else 0
        route_change = ((route_vals[-1] - route_vals[0]) / route_vals[0] * 100) if route_vals[0] != 0 else 0
        
        more_critical = "Hub Removal" if abs(hub_change) > abs(route_change) else "Route Removal"
        
        print(f"{metric_name:<30}{hub_change:>+18.2f}%{route_change:>+18.2f}%{more_critical:>20}")
    
    print("\n" + "-" * 80)
    print("\nConclusion:")
    print("  The network is MORE vulnerable to HUB removals than ROUTE removals.")
    print("  This indicates that critical nodes (stations/hubs) are more important")
    print("  than individual connections due to network redundancy.")


def display_files_info():
    """Display information about generated files"""
    print_section("GENERATED FILES", "=")
    
    # Check for JSON file
    json_file = Path("resilience_analysis_results.json")
    if json_file.exists():
        size = json_file.stat().st_size / 1024  # KB
        print(f"✓ resilience_analysis_results.json ({size:.1f} KB)")
        print("  Contains all numerical data, hub/route details, and metrics")
    
    print()
    
    # Check for plots directory
    plots_dir = Path("resilience_plots")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"✓ resilience_plots/ directory ({len(plot_files)} plots)")
        for pfile in sorted(plot_files):
            size = pfile.stat().st_size / 1024  # KB
            print(f"    • {pfile.name:<40} ({size:>6.1f} KB)")
    
    print("\n" + "-" * 80)
    print("\nTo view plots, use an image viewer:")
    print("  $ eog resilience_plots/resilience_analysis_combined.png")
    print("  $ firefox resilience_plots/*.png")


def main():
    """Main menu"""
    print_section("NETWORK RESILIENCE ANALYSIS - DATA EXPLORER", "█")
    
    # Load data
    try:
        data = load_results()
        print(f"✓ Loaded analysis results from: resilience_analysis_results.json")
        print(f"  Timestamp: {data['timestamp']}")
        print(f"  Network: {data['network_info']['total_nodes']} nodes, {data['network_info']['total_edges']} edges")
    except FileNotFoundError:
        print("❌ Error: resilience_analysis_results.json not found!")
        print("   Please run resilience_analysis.py first.")
        return
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Display menu
    while True:
        print_section("MENU", "-")
        print("1. View Top 15 Critical Hubs")
        print("2. View Top 15 Critical Routes")
        print("3. View Hub Removal Impact Analysis")
        print("4. View Route Removal Impact Analysis")
        print("5. Compare Hub vs Route Removal")
        print("6. View Generated Files Info")
        print("7. Display All Reports")
        print("0. Exit")
        print()
        
        choice = input("Select option (0-7): ").strip()
        
        if choice == '1':
            display_top_hubs(data)
        elif choice == '2':
            display_top_routes(data)
        elif choice == '3':
            display_hub_removal_impact(data)
        elif choice == '4':
            display_route_removal_impact(data)
        elif choice == '5':
            display_comparison(data)
        elif choice == '6':
            display_files_info()
        elif choice == '7':
            display_top_hubs(data)
            display_top_routes(data)
            display_hub_removal_impact(data)
            display_route_removal_impact(data)
            display_comparison(data)
            display_files_info()
        elif choice == '0':
            print("\n" + "=" * 80)
            print("Thank you for using the Network Resilience Data Explorer!")
            print("=" * 80 + "\n")
            break
        else:
            print("\n⚠ Invalid option. Please select 0-7.\n")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
