#!/usr/bin/env python3
"""
Multilayer Network Resilience Analysis
Analyzes the impact of removing major hubs and routes from Hyderabad's transport network
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MultiLayerNetworkAnalyzer:
    """Analyzes resilience of multilayer transport network"""
    
    def __init__(self, nodes_file: str, edges_file: str):
        """Initialize with network data files"""
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.G = None
        self._build_graph()
        
    def _build_graph(self):
        """Build NetworkX graph from CSV data"""
        self.G = nx.Graph()
        
        # Add nodes with attributes
        for _, node in self.nodes_df.iterrows():
            self.G.add_node(
                node['node_id'],
                name=node['name'],
                lat=node['lat'],
                lon=node['lon'],
                layer=node['layer'],
                type=node['type'],
                region=node['region']
            )
        
        # Add edges with attributes
        for _, edge in self.edges_df.iterrows():
            self.G.add_edge(
                edge['from_id'],
                edge['to_id'],
                mode=edge['mode'],
                distance=edge['distance_km'],
                time=edge['time_min'],
                cost=edge['cost_rs'],
                region_from=edge['region_from'],
                region_to=edge['region_to']
            )
        
        print(f"✓ Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def calculate_network_metrics(self, G: nx.Graph = None) -> Dict[str, float]:
        """Calculate comprehensive network metrics"""
        if G is None:
            G = self.G
        
        if G.number_of_nodes() == 0:
            return {
                'avg_clustering': 0.0,
                'avg_betweenness': 0.0,
                'avg_closeness': 0.0,
                'avg_degree': 0.0,
                'avg_time': 0.0,
                'avg_cost': 0.0,
                'num_components': 0,
                'largest_component_size': 0,
                'network_efficiency': 0.0
            }
        
        # Get largest connected component
        if nx.is_connected(G):
            largest_cc = G
        else:
            largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        # Calculate centrality metrics on largest component
        clustering_dict = nx.clustering(largest_cc)
        avg_clustering = np.mean(list(clustering_dict.values())) if clustering_dict else 0.0
        
        # Betweenness centrality (normalized)
        betweenness_dict = nx.betweenness_centrality(largest_cc, normalized=True)
        avg_betweenness = np.mean(list(betweenness_dict.values())) if betweenness_dict else 0.0
        
        # Closeness centrality (normalized)
        closeness_dict = nx.closeness_centrality(largest_cc)
        avg_closeness = np.mean(list(closeness_dict.values())) if closeness_dict else 0.0
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0.0
        
        # Time and cost statistics
        times = [data['time'] for u, v, data in G.edges(data=True) if 'time' in data]
        costs = [data['cost'] for u, v, data in G.edges(data=True) if 'cost' in data]
        
        avg_time = np.mean(times) if times else 0.0
        avg_cost = np.mean(costs) if costs else 0.0
        
        # Network efficiency (average inverse shortest path length)
        efficiency = 0.0
        if largest_cc.number_of_nodes() > 1:
            try:
                efficiency = nx.global_efficiency(largest_cc)
            except:
                efficiency = 0.0
        
        return {
            'avg_clustering': avg_clustering,
            'avg_betweenness': avg_betweenness,
            'avg_closeness': avg_closeness,
            'avg_degree': avg_degree,
            'avg_time': avg_time,
            'avg_cost': avg_cost,
            'num_components': nx.number_connected_components(G),
            'largest_component_size': largest_cc.number_of_nodes(),
            'network_efficiency': efficiency
        }
    
    def identify_top_hubs_by_metric(self, metric: str, top_n: int = 15) -> List[Tuple[str, float, Dict]]:
        """
        Identify top N hubs based on a SINGLE centrality metric
        For multilayer network analysis across ALL layers
        
        Args:
            metric: 'degree', 'betweenness', or 'closeness'
            top_n: Number of top hubs to identify
        
        Returns: List of (node_id, metric_value, metrics_dict)
        """
        print(f"\n🔍 Identifying top {top_n} hubs by {metric.upper()}...")
        
        if metric == 'degree':
            print("  📊 Calculating degree centrality...")
            centrality = nx.degree_centrality(self.G)
            degree_dict = dict(self.G.degree())
            
        elif metric == 'betweenness':
            print("  🔀 Calculating betweenness centrality (unweighted for multilayer balance)...")
            centrality = nx.betweenness_centrality(self.G, normalized=True)
            degree_dict = dict(self.G.degree())
            
        elif metric == 'closeness':
            print("  📍 Calculating closeness centrality (unweighted for multilayer balance)...")
            centrality = nx.closeness_centrality(self.G)
            degree_dict = dict(self.G.degree())
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Build results
        print("  🎯 Ranking nodes...")
        results = {}
        for node in self.G.nodes():
            results[node] = {
                f'{metric}_centrality': centrality[node],
                'degree': degree_dict.get(node, 0),
                'layer': self.G.nodes[node].get('layer', 'unknown'),
                'name': self.G.nodes[node].get('name', node)
            }
        
        # Sort by metric value
        sorted_nodes = sorted(
            results.items(),
            key=lambda x: x[1][f'{metric}_centrality'],
            reverse=True
        )
        
        top_hubs = [(node, data[f'{metric}_centrality'], data) for node, data in sorted_nodes[:top_n]]
        
        # Print top hubs
        print(f"\n{'Rank':<6}{'Node ID':<20}{'Name':<30}{'Layer':<10}{metric.title():<12}{'Degree':<8}")
        print("=" * 96)
        for i, (node_id, value, data) in enumerate(top_hubs, 1):
            print(f"{i:<6}{node_id:<20}{data['name']:<30}{data['layer']:<10}{value:<12.4f}{data['degree']:<8}")
        
        return top_hubs
    
    def identify_top_hubs(self, top_n: int = 15) -> List[Tuple[str, float, Dict]]:
        """
        Identify top N hubs based on UNWEIGHTED composite score
        For multilayer network - considers structural importance across ALL layers
        Returns: List of (node_id, composite_score, metrics_dict)
        """
        print(f"\n🔍 Identifying top {top_n} hubs (COMPOSITE SCORE - Multilayer Network)...")
        
        # Calculate various centrality metrics (UNWEIGHTED for multilayer balance)
        print("  � Calculating degree centrality...")
        degree_cent = nx.degree_centrality(self.G)
        
        print("  🔀 Calculating betweenness centrality (unweighted)...")
        betweenness_cent = nx.betweenness_centrality(self.G, normalized=True)
        
        print("  📍 Calculating closeness centrality (unweighted)...")
        closeness_cent = nx.closeness_centrality(self.G)
        
        # PageRank for importance
        try:
            print("  🔗 Calculating PageRank...")
            pagerank = nx.pagerank(self.G, alpha=0.85)
        except:
            pagerank = {node: 1.0/self.G.number_of_nodes() for node in self.G.nodes()}
        
        # Eigenvector centrality
        try:
            print("  🌐 Calculating eigenvector centrality...")
            eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=1000)
        except:
            eigenvector_cent = {node: 0.0 for node in self.G.nodes()}
        
        # Composite scoring (weighted combination)
        print("  🎯 Computing composite scores...")
        composite_scores = {}
        for node in self.G.nodes():
            score = (
                0.25 * degree_cent.get(node, 0) +
                0.30 * betweenness_cent.get(node, 0) +
                0.20 * closeness_cent.get(node, 0) +
                0.15 * pagerank.get(node, 0) +
                0.10 * eigenvector_cent.get(node, 0)
            )
            composite_scores[node] = {
                'composite_score': score,
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': betweenness_cent[node],
                'closeness_centrality': closeness_cent[node],
                'pagerank': pagerank[node],
                'eigenvector_centrality': eigenvector_cent.get(node, 0),
                'degree': self.G.degree(node),
                'layer': self.G.nodes[node].get('layer', 'unknown'),
                'name': self.G.nodes[node].get('name', node)
            }
        
        # Sort by composite score
        sorted_nodes = sorted(
            composite_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        top_hubs = [(node, data['composite_score'], data) for node, data in sorted_nodes[:top_n]]
        
        # Print top hubs
        print(f"\n{'Rank':<6}{'Node ID':<20}{'Name':<30}{'Layer':<10}{'Score':<10}{'Degree':<8}")
        print("=" * 94)
        for i, (node_id, score, data) in enumerate(top_hubs, 1):
            print(f"{i:<6}{node_id:<20}{data['name']:<30}{data['layer']:<10}{score:<10.4f}{data['degree']:<8}")
        
        return top_hubs
    
    def identify_top_routes(self, top_n: int = 15) -> List[Tuple[Tuple[str, str], float, Dict]]:
        """
        Identify top N critical routes/edges based on UNWEIGHTED edge betweenness
        For multilayer network - considers structural importance across ALL layers
        Returns: List of ((from_id, to_id), score, metrics_dict)
        """
        print(f"\n🛣️  Identifying top {top_n} critical routes...")
        
        # UNWEIGHTED Edge betweenness centrality for multilayer balance
        print("  🔀 Calculating edge betweenness (unweighted)...")
        edge_betweenness = nx.edge_betweenness_centrality(self.G, normalized=True)
        
        # Calculate route scores
        print("  🎯 Computing route importance scores...")
        route_scores = {}
        for (u, v), betweenness in edge_betweenness.items():
            edge_data = self.G[u][v]
            
            # Score based on betweenness and endpoint degrees
            degree_factor = (self.G.degree(u) + self.G.degree(v)) / 2
            max_degree = max(dict(self.G.degree()).values())
            
            score = (
                0.70 * betweenness +                               # Primary factor - path criticality
                0.30 * (degree_factor / max_degree)                # Secondary - hub connectivity
            )
            
            route_scores[(u, v)] = {
                'score': score,
                'edge_betweenness': betweenness,
                'mode': edge_data.get('mode', 'unknown'),
                'distance_km': edge_data.get('distance', 0),
                'time_min': edge_data.get('time', 0),
                'cost_rs': edge_data.get('cost', 0),
                'from_name': self.G.nodes[u].get('name', u),
                'to_name': self.G.nodes[v].get('name', v),
                'region_from': edge_data.get('region_from', ''),
                'region_to': edge_data.get('region_to', '')
            }
        
        # Sort by score
        sorted_routes = sorted(
            route_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_routes = [(edge, data['score'], data) for edge, data in sorted_routes[:top_n]]
        
        # Print top routes
        print(f"\n{'Rank':<6}{'From':<25}{'To':<25}{'Mode':<10}{'Score':<10}{'Betw.':<10}")
        print("=" * 96)
        for i, ((u, v), score, data) in enumerate(top_routes, 1):
            print(f"{i:<6}{data['from_name']:<25}{data['to_name']:<25}{data['mode']:<10}{score:<10.4f}{data['edge_betweenness']:<10.4f}")
        
        return top_routes
    
    def analyze_hub_removal(self, hub_id: str) -> Dict[str, Any]:
        """Analyze network metrics after removing a hub"""
        # Create graph without this hub
        G_removed = self.G.copy()
        G_removed.remove_node(hub_id)
        
        # Calculate metrics
        metrics = self.calculate_network_metrics(G_removed)
        
        # Additional impact metrics
        metrics['removed_hub'] = hub_id
        metrics['nodes_remaining'] = G_removed.number_of_nodes()
        metrics['edges_remaining'] = G_removed.number_of_edges()
        
        return metrics
    
    def analyze_route_removal(self, route: Tuple[str, str]) -> Dict[str, Any]:
        """Analyze network metrics after removing a route"""
        # Create graph without this route
        G_removed = self.G.copy()
        u, v = route
        
        # Remove edge in both directions if they exist
        if G_removed.has_edge(u, v):
            G_removed.remove_edge(u, v)
        if G_removed.has_edge(v, u):
            G_removed.remove_edge(v, u)
        
        # Calculate metrics
        metrics = self.calculate_network_metrics(G_removed)
        
        # Additional impact metrics
        metrics['removed_route'] = f"{u} -> {v}"
        metrics['nodes_remaining'] = G_removed.number_of_nodes()
        metrics['edges_remaining'] = G_removed.number_of_edges()
        
        return metrics
    
    def sequential_hub_removal_analysis(self, top_hubs: List[Tuple[str, float, Dict]]) -> Dict[str, List]:
        """
        Sequentially remove hubs and analyze network impact
        Returns: Dictionary with metrics tracked across removals
        """
        print(f"\n📊 Sequential Hub Removal Analysis...")
        
        results = {
            'removal_sequence': [],
            'avg_clustering': [],
            'avg_betweenness': [],
            'avg_closeness': [],
            'avg_degree': [],
            'avg_time': [],
            'avg_cost': [],
            'num_components': [],
            'largest_component_size': [],
            'network_efficiency': [],
            'hub_details': []
        }
        
        # Baseline metrics
        baseline_metrics = self.calculate_network_metrics()
        print(f"\n📈 Baseline Network Metrics:")
        for key, value in baseline_metrics.items():
            print(f"   {key}: {value:.4f}")
        
        # Create a working copy of the graph
        G_working = self.G.copy()
        
        # Sequentially remove hubs
        for i, (hub_id, score, hub_data) in enumerate(top_hubs, 1):
            print(f"\n🔻 Removing hub {i}/15: {hub_data['name']} ({hub_id})")
            
            # Remove hub
            if hub_id in G_working:
                G_working.remove_node(hub_id)
            
            # Calculate metrics
            metrics = self.calculate_network_metrics(G_working)
            
            # Store results
            results['removal_sequence'].append(i)
            results['avg_clustering'].append(metrics['avg_clustering'])
            results['avg_betweenness'].append(metrics['avg_betweenness'])
            results['avg_closeness'].append(metrics['avg_closeness'])
            results['avg_degree'].append(metrics['avg_degree'])
            results['avg_time'].append(metrics['avg_time'])
            results['avg_cost'].append(metrics['avg_cost'])
            results['num_components'].append(metrics['num_components'])
            results['largest_component_size'].append(metrics['largest_component_size'])
            results['network_efficiency'].append(metrics['network_efficiency'])
            results['hub_details'].append({
                'rank': i,
                'hub_id': hub_id,
                'name': hub_data['name'],
                'layer': hub_data['layer'],
                'original_degree': hub_data['degree'],
                'composite_score': score
            })
            
            print(f"   Remaining nodes: {G_working.number_of_nodes()}, edges: {G_working.number_of_edges()}")
            print(f"   Components: {metrics['num_components']}, Largest: {metrics['largest_component_size']}")
            print(f"   Efficiency: {metrics['network_efficiency']:.4f}")
        
        return results
    
    def sequential_route_removal_analysis(self, top_routes: List[Tuple[Tuple[str, str], float, Dict]]) -> Dict[str, List]:
        """
        Sequentially remove routes and analyze network impact
        Returns: Dictionary with metrics tracked across removals
        """
        print(f"\n📊 Sequential Route Removal Analysis...")
        
        results = {
            'removal_sequence': [],
            'avg_clustering': [],
            'avg_betweenness': [],
            'avg_closeness': [],
            'avg_degree': [],
            'avg_time': [],
            'avg_cost': [],
            'num_components': [],
            'largest_component_size': [],
            'network_efficiency': [],
            'route_details': []
        }
        
        # Create a working copy of the graph
        G_working = self.G.copy()
        
        # Sequentially remove routes
        for i, ((u, v), score, route_data) in enumerate(top_routes, 1):
            print(f"\n🔻 Removing route {i}/15: {route_data['from_name']} -> {route_data['to_name']}")
            
            # Remove route (both directions)
            if G_working.has_edge(u, v):
                G_working.remove_edge(u, v)
            if G_working.has_edge(v, u):
                G_working.remove_edge(v, u)
            
            # Calculate metrics
            metrics = self.calculate_network_metrics(G_working)
            
            # Store results
            results['removal_sequence'].append(i)
            results['avg_clustering'].append(metrics['avg_clustering'])
            results['avg_betweenness'].append(metrics['avg_betweenness'])
            results['avg_closeness'].append(metrics['avg_closeness'])
            results['avg_degree'].append(metrics['avg_degree'])
            results['avg_time'].append(metrics['avg_time'])
            results['avg_cost'].append(metrics['avg_cost'])
            results['num_components'].append(metrics['num_components'])
            results['largest_component_size'].append(metrics['largest_component_size'])
            results['network_efficiency'].append(metrics['network_efficiency'])
            results['route_details'].append({
                'rank': i,
                'from_id': u,
                'to_id': v,
                'from_name': route_data['from_name'],
                'to_name': route_data['to_name'],
                'mode': route_data['mode'],
                'distance_km': route_data['distance_km'],
                'time_min': route_data['time_min'],
                'score': score
            })
            
            print(f"   Remaining edges: {G_working.number_of_edges()}")
            print(f"   Components: {metrics['num_components']}, Largest: {metrics['largest_component_size']}")
            print(f"   Efficiency: {metrics['network_efficiency']:.4f}")
        
        return results


def plot_single_removal_analysis(results: Dict, analysis_name: str, output_dir: str = "."):
    """
    Create comprehensive visualization plots for a SINGLE removal analysis
    
    Args:
        results: Dictionary with removal analysis results
        analysis_name: Name of the analysis (e.g., "Degree Removal", "Route Removal")
        output_dir: Directory to save plots
    """
    print(f"\n📊 Creating plots for {analysis_name}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot
    metrics = [
        ('avg_clustering', 'Average Clustering Coefficient', '#e74c3c'),
        ('avg_betweenness', 'Average Betweenness Centrality', '#3498db'),
        ('avg_closeness', 'Average Closeness Centrality', '#2ecc71'),
        ('avg_degree', 'Average Node Degree', '#f39c12'),
        ('avg_time', 'Average Travel Time (min)', '#9b59b6'),
        ('avg_cost', 'Average Travel Cost (₹)', '#1abc9c'),
        ('network_efficiency', 'Network Efficiency', '#e67e22'),
        ('num_components', 'Number of Connected Components', '#34495e'),
        ('largest_component_size', 'Largest Component Size', '#16a085')
    ]
    
    # Create individual plots for each metric
    for metric_key, metric_name, color in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(results['removal_sequence'], results[metric_key], 
                marker='o', linewidth=2.5, markersize=10, color=color, 
                label=analysis_name, alpha=0.8)
        ax.set_xlabel('Removal Sequence (1st → 15th)', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_name}\n{analysis_name} Impact', fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(results['removal_sequence'])
        ax.legend(fontsize=11, loc='best')
        
        plt.tight_layout()
        filename = f"{metric_key}.png"
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
        plt.close()
    
    # Create combined dashboard (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()
    fig.suptitle(f'{analysis_name} - Network Resilience Dashboard', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    for idx, (metric_key, metric_name, color) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(results['removal_sequence'], results[metric_key],
                marker='o', linewidth=2, markersize=8, color=color, alpha=0.8)
        ax.set_xlabel('Removal Sequence', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xticks(results['removal_sequence'])
    
    plt.tight_layout()
    filename = "combined_dashboard.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()
    
    print(f"   ✓ All plots saved to: {output_dir}/")


def plot_resilience_analysis(hub_results: Dict, route_results: Dict, output_dir: str = "."):
    """
    Create comprehensive visualization plots for resilience analysis
    """
    print(f"\n📊 Creating visualization plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Metrics to plot
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
    
    # Create individual plots for each metric
    for metric_key, metric_name in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Hub removal plot
        ax1.plot(hub_results['removal_sequence'], hub_results[metric_key], 
                marker='o', linewidth=2, markersize=8, color='#e74c3c', label='Hub Removal')
        ax1.set_xlabel('Hub Removal Sequence (1st → 15th)', fontsize=12, fontweight='bold')
        ax1.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax1.set_title(f'{metric_name} - Hub Removal Impact', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(hub_results['removal_sequence'])
        
        # Route removal plot
        ax2.plot(route_results['removal_sequence'], route_results[metric_key],
                marker='s', linewidth=2, markersize=8, color='#3498db', label='Route Removal')
        ax2.set_xlabel('Route Removal Sequence (1st → 15th)', fontsize=12, fontweight='bold')
        ax2.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax2.set_title(f'{metric_name} - Route Removal Impact', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(route_results['removal_sequence'])
        
        plt.tight_layout()
        filename = f"{metric_key}_resilience.png"
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
        plt.close()
    
    # Combined comparison plot (all metrics normalized)
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        # Normalize values to 0-1 range for comparison
        hub_vals = np.array(hub_results[metric_key])
        route_vals = np.array(route_results[metric_key])
        
        # Handle cases where max might be 0
        hub_max = hub_vals.max() if hub_vals.max() > 0 else 1
        route_max = route_vals.max() if route_vals.max() > 0 else 1
        
        hub_normalized = hub_vals / hub_max
        route_normalized = route_vals / route_max
        
        ax.plot(hub_results['removal_sequence'], hub_normalized, 
               marker='o', linewidth=2, markersize=6, color='#e74c3c', label='Hub Removal')
        ax.plot(route_results['removal_sequence'], route_normalized,
               marker='s', linewidth=2, markersize=6, color='#3498db', label='Route Removal')
        
        ax.set_xlabel('Removal Sequence', fontsize=10)
        ax.set_ylabel('Normalized Value', fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xticks(hub_results['removal_sequence'])
    
    plt.suptitle('Network Resilience Analysis - Comparative Overview', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / "resilience_analysis_combined.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: resilience_analysis_combined.png")
    plt.close()
    
    print(f"\n✅ All plots saved to: {output_path.absolute()}")


def main():
    """
    Main execution function - Runs 4 SEPARATE removal analyses:
    1. Remove top 15 nodes by DEGREE centrality
    2. Remove top 15 nodes by BETWEENNESS centrality  
    3. Remove top 15 nodes by CLOSENESS centrality
    4. Remove top 15 ROUTES
    """
    print("=" * 90)
    print("MULTILAYER NETWORK RESILIENCE ANALYSIS")
    print("Hyderabad Urban Mobility & Public Transport Network")
    print("4 SEPARATE REMOVAL STRATEGIES")
    print("=" * 90)
    
    # File paths
    nodes_file = "nodes.csv"
    edges_file = "edges.csv"
    
    # Initialize analyzer
    analyzer = MultiLayerNetworkAnalyzer(nodes_file, edges_file)
    
    # Network info (same for all analyses)
    network_info = {
        'total_nodes': analyzer.G.number_of_nodes(),
        'total_edges': analyzer.G.number_of_edges(),
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # ========================================================================
    # ANALYSIS 1: Remove by DEGREE Centrality
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 1: TOP 15 NODES BY DEGREE CENTRALITY")
    print("=" * 90)
    
    top_hubs_degree = analyzer.identify_top_hubs_by_metric('degree', top_n=15)
    degree_removal_results = analyzer.sequential_hub_removal_analysis(top_hubs_degree)
    
    # Save results
    degree_output = {
        'analysis_type': 'degree_centrality_removal',
        'network_info': network_info,
        'top_hubs': [
            {'rank': i, 'node_id': node_id, 'degree_centrality': score, **metrics}
            for i, (node_id, score, metrics) in enumerate(top_hubs_degree, 1)
        ],
        'removal_analysis': degree_removal_results
    }
    
    degree_dir = Path('degree_removal')
    degree_dir.mkdir(exist_ok=True)
    with open(degree_dir / 'results.json', 'w') as f:
        json.dump(degree_output, f, indent=2)
    print(f"\n💾 Results saved to: degree_removal/results.json")
    
    # Create visualizations
    plot_single_removal_analysis(degree_removal_results, "Degree Centrality Removal", 
                                 output_dir=str(degree_dir))
    
    # ========================================================================
    # ANALYSIS 2: Remove by BETWEENNESS Centrality
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 2: TOP 15 NODES BY BETWEENNESS CENTRALITY")
    print("=" * 90)
    
    top_hubs_betweenness = analyzer.identify_top_hubs_by_metric('betweenness', top_n=15)
    betweenness_removal_results = analyzer.sequential_hub_removal_analysis(top_hubs_betweenness)
    
    # Save results
    betweenness_output = {
        'analysis_type': 'betweenness_centrality_removal',
        'network_info': network_info,
        'top_hubs': [
            {'rank': i, 'node_id': node_id, 'betweenness_centrality': score, **metrics}
            for i, (node_id, score, metrics) in enumerate(top_hubs_betweenness, 1)
        ],
        'removal_analysis': betweenness_removal_results
    }
    
    betweenness_dir = Path('betweenness_removal')
    betweenness_dir.mkdir(exist_ok=True)
    with open(betweenness_dir / 'results.json', 'w') as f:
        json.dump(betweenness_output, f, indent=2)
    print(f"\n💾 Results saved to: betweenness_removal/results.json")
    
    # Create visualizations
    plot_single_removal_analysis(betweenness_removal_results, "Betweenness Centrality Removal",
                                 output_dir=str(betweenness_dir))
    
    # ========================================================================
    # ANALYSIS 3: Remove by CLOSENESS Centrality
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 3: TOP 15 NODES BY CLOSENESS CENTRALITY")
    print("=" * 90)
    
    top_hubs_closeness = analyzer.identify_top_hubs_by_metric('closeness', top_n=15)
    closeness_removal_results = analyzer.sequential_hub_removal_analysis(top_hubs_closeness)
    
    # Save results
    closeness_output = {
        'analysis_type': 'closeness_centrality_removal',
        'network_info': network_info,
        'top_hubs': [
            {'rank': i, 'node_id': node_id, 'closeness_centrality': score, **metrics}
            for i, (node_id, score, metrics) in enumerate(top_hubs_closeness, 1)
        ],
        'removal_analysis': closeness_removal_results
    }
    
    closeness_dir = Path('closeness_removal')
    closeness_dir.mkdir(exist_ok=True)
    with open(closeness_dir / 'results.json', 'w') as f:
        json.dump(closeness_output, f, indent=2)
    print(f"\n💾 Results saved to: closeness_removal/results.json")
    
    # Create visualizations
    plot_single_removal_analysis(closeness_removal_results, "Closeness Centrality Removal",
                                 output_dir=str(closeness_dir))
    
    # ========================================================================
    # ANALYSIS 4: Remove TOP 15 ROUTES
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 4: TOP 15 CRITICAL ROUTES")
    print("=" * 90)
    
    top_routes = analyzer.identify_top_routes(top_n=15)
    route_removal_results = analyzer.sequential_route_removal_analysis(top_routes)
    
    # Save results
    route_output = {
        'analysis_type': 'route_removal',
        'network_info': network_info,
        'top_routes': [
            {'rank': i, 'from_id': u, 'to_id': v, 'score': score, **metrics}
            for i, ((u, v), score, metrics) in enumerate(top_routes, 1)
        ],
        'removal_analysis': route_removal_results
    }
    
    route_dir = Path('route_removal')
    route_dir.mkdir(exist_ok=True)
    with open(route_dir / 'results.json', 'w') as f:
        json.dump(route_output, f, indent=2)
    print(f"\n💾 Results saved to: route_removal/results.json")
    
    # Create visualizations
    plot_single_removal_analysis(route_removal_results, "Route Removal",
                                 output_dir=str(route_dir))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 90)
    print("✅ ALL 4 ANALYSES COMPLETE!")
    print("=" * 90)
    print("\nGenerated directories and files:")
    print("\n  1. degree_removal/")
    print("       • results.json - Top 15 nodes by degree + removal analysis")
    print("       • 10 visualization plots (.png files)")
    print("\n  2. betweenness_removal/")
    print("       • results.json - Top 15 nodes by betweenness + removal analysis")
    print("       • 10 visualization plots (.png files)")
    print("\n  3. closeness_removal/")
    print("       • results.json - Top 15 nodes by closeness + removal analysis")
    print("       • 10 visualization plots (.png files)")
    print("\n  4. route_removal/")
    print("       • results.json - Top 15 routes + removal analysis")
    print("       • 10 visualization plots (.png files)")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
