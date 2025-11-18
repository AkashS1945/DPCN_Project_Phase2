#!/usr/bin/env python3
"""
Complete Network Analysis - Determine Network Type and Properties
Analyzes Hyderabad transport network to identify its type (small-world, scale-free, etc.)
and calculate all relevant graph properties
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

class NetworkTypeAnalyzer:
    """Comprehensive network type and property analyzer"""
    
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
                name=node.get('name', ''),
                lat=node.get('lat', 0),
                lon=node.get('lon', 0),
                layer=node.get('layer', ''),
                type=node.get('type', ''),
                region=node.get('region', '')
            )
        
        # Add edges with attributes - handle directed edges (count only once)
        edge_set = set()
        for _, edge in self.edges_df.iterrows():
            edge_tuple = tuple(sorted([edge['from_id'], edge['to_id']]))
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                self.G.add_edge(
                    edge['from_id'],
                    edge['to_id'],
                    mode=edge.get('mode', ''),
                    distance=edge.get('distance_km', 0),
                    time=edge.get('time_min', 0),
                    cost=edge.get('cost_rs', 0)
                )
        
        print(f"✓ Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"  (Original edges CSV has {len(self.edges_df)} entries - likely bidirectional)")
    
    def analyze_complete_network(self) -> dict:
        """Perform comprehensive network analysis"""
        
        analysis = {
            'basic_properties': self._basic_properties(),
            'connectivity': self._connectivity_analysis(),
            'centrality': self._centrality_analysis(),
            'clustering': self._clustering_analysis(),
            'path_analysis': self._path_analysis(),
            'degree_distribution': self._degree_distribution_analysis(),
            'network_type': self._identify_network_type(),
            'resilience': self._resilience_metrics(),
            'efficiency': self._efficiency_metrics(),
            'community': self._community_analysis(),
            'spatial': self._spatial_analysis()
        }
        
        return analysis
    
    def _basic_properties(self) -> dict:
        """Calculate basic network properties"""
        return {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'is_connected': nx.is_connected(self.G),
            'num_components': nx.number_connected_components(self.G),
            'is_directed': self.G.is_directed(),
            'is_weighted': nx.is_weighted(self.G, weight='distance'),
            'is_planar': nx.check_planarity(self.G)[0] if self.G.number_of_nodes() < 1000 else None
        }
    
    def _connectivity_analysis(self) -> dict:
        """Analyze network connectivity"""
        components = list(nx.connected_components(self.G))
        largest_cc = max(components, key=len)
        
        # Node connectivity (minimum number of nodes to disconnect)
        node_connectivity = nx.node_connectivity(self.G)
        
        # Edge connectivity (minimum number of edges to disconnect)
        edge_connectivity = nx.edge_connectivity(self.G)
        
        return {
            'num_components': len(components),
            'largest_component_size': len(largest_cc),
            'largest_component_fraction': len(largest_cc) / self.G.number_of_nodes(),
            'node_connectivity': node_connectivity,
            'edge_connectivity': edge_connectivity,
            'is_biconnected': nx.is_biconnected(self.G),
            'num_articulation_points': len(list(nx.articulation_points(self.G))),
            'num_bridges': sum(1 for _ in nx.bridges(self.G))
        }
    
    def _centrality_analysis(self) -> dict:
        """Calculate centrality metrics"""
        # Get largest component for metrics that require connected graph
        if nx.is_connected(self.G):
            G_conn = self.G
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_conn = self.G.subgraph(largest_cc).copy()
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.G)
        
        # Betweenness centrality
        betweenness_cent = nx.betweenness_centrality(G_conn)
        
        # Closeness centrality
        closeness_cent = nx.closeness_centrality(G_conn)
        
        # Eigenvector centrality
        try:
            eigenvector_cent = nx.eigenvector_centrality(G_conn, max_iter=1000)
        except:
            eigenvector_cent = {}
        
        # PageRank
        pagerank = nx.pagerank(self.G)
        
        return {
            'avg_degree_centrality': np.mean(list(degree_cent.values())),
            'max_degree_centrality': max(degree_cent.values()),
            'avg_betweenness_centrality': np.mean(list(betweenness_cent.values())),
            'max_betweenness_centrality': max(betweenness_cent.values()),
            'avg_closeness_centrality': np.mean(list(closeness_cent.values())),
            'max_closeness_centrality': max(closeness_cent.values()),
            'avg_eigenvector_centrality': np.mean(list(eigenvector_cent.values())) if eigenvector_cent else 0,
            'avg_pagerank': np.mean(list(pagerank.values())),
            'top_5_degree_nodes': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_5_betweenness_nodes': sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _clustering_analysis(self) -> dict:
        """Analyze clustering properties"""
        clustering_coeff = nx.clustering(self.G)
        avg_clustering = nx.average_clustering(self.G)
        
        # Transitivity (global clustering coefficient)
        transitivity = nx.transitivity(self.G)
        
        return {
            'avg_clustering_coefficient': avg_clustering,
            'transitivity': transitivity,
            'max_clustering_coefficient': max(clustering_coeff.values()) if clustering_coeff else 0,
            'min_clustering_coefficient': min(clustering_coeff.values()) if clustering_coeff else 0
        }
    
    def _path_analysis(self) -> dict:
        """Analyze path properties"""
        # Get largest connected component
        if nx.is_connected(self.G):
            G_conn = self.G
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_conn = self.G.subgraph(largest_cc).copy()
        
        # Average shortest path length
        avg_shortest_path = nx.average_shortest_path_length(G_conn)
        
        # Diameter (longest shortest path)
        diameter = nx.diameter(G_conn)
        
        # Radius (minimum eccentricity)
        radius = nx.radius(G_conn)
        
        # Eccentricity
        eccentricity = nx.eccentricity(G_conn)
        
        return {
            'avg_shortest_path_length': avg_shortest_path,
            'diameter': diameter,
            'radius': radius,
            'avg_eccentricity': np.mean(list(eccentricity.values())),
            'max_eccentricity': max(eccentricity.values()),
            'center_size': len(nx.center(G_conn)),
            'periphery_size': len(nx.periphery(G_conn))
        }
    
    def _degree_distribution_analysis(self) -> dict:
        """Analyze degree distribution"""
        degrees = [d for n, d in self.G.degree()]
        degree_counts = Counter(degrees)
        
        # Statistics
        avg_degree = np.mean(degrees)
        median_degree = np.median(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
        std_degree = np.std(degrees)
        
        # Power-law test (for scale-free networks)
        # Fit power law: P(k) ~ k^(-gamma)
        degree_values = np.array(list(degree_counts.keys()))
        degree_probs = np.array(list(degree_counts.values())) / sum(degree_counts.values())
        
        # Filter out zeros for log-log regression
        mask = (degree_values > 0) & (degree_probs > 0)
        if mask.sum() > 2:
            log_k = np.log(degree_values[mask])
            log_pk = np.log(degree_probs[mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_pk)
            power_law_exponent = -slope
            power_law_r_squared = r_value ** 2
        else:
            power_law_exponent = None
            power_law_r_squared = None
        
        return {
            'avg_degree': avg_degree,
            'median_degree': median_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'std_degree': std_degree,
            'degree_assortativity': nx.degree_assortativity_coefficient(self.G),
            'power_law_exponent': power_law_exponent,
            'power_law_fit_r_squared': power_law_r_squared,
            'degree_distribution': dict(degree_counts)
        }
    
    def _identify_network_type(self) -> dict:
        """Identify network type (small-world, scale-free, random, etc.)"""
        
        # Get basic properties
        if nx.is_connected(self.G):
            G_conn = self.G
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_conn = self.G.subgraph(largest_cc).copy()
        
        n = G_conn.number_of_nodes()
        m = G_conn.number_of_edges()
        avg_degree = 2 * m / n
        
        C = nx.average_clustering(G_conn)
        L = nx.average_shortest_path_length(G_conn)
        
        # Generate random graph with same n and m for comparison
        G_random = nx.gnm_random_graph(n, m)
        C_random = nx.average_clustering(G_random)
        L_random = nx.average_shortest_path_length(G_random) if nx.is_connected(G_random) else L
        
        # Small-world criteria:
        # C >> C_random (high clustering)
        # L ≈ L_random (small path length)
        # Typically: C/C_random > 3 and L/L_random < 2
        
        clustering_ratio = C / C_random if C_random > 0 else 0
        path_ratio = L / L_random if L_random > 0 else 0
        
        is_small_world = clustering_ratio > 3 and path_ratio < 2
        
        # Scale-free test: power-law degree distribution
        # Typically gamma between 2 and 3
        degrees = [d for n, d in G_conn.degree()]
        degree_counts = Counter(degrees)
        
        degree_values = np.array(list(degree_counts.keys()))
        degree_probs = np.array(list(degree_counts.values())) / sum(degree_counts.values())
        
        mask = (degree_values > 0) & (degree_probs > 0)
        if mask.sum() > 2:
            log_k = np.log(degree_values[mask])
            log_pk = np.log(degree_probs[mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_pk)
            gamma = -slope
            r_squared = r_value ** 2
            is_scale_free = (2 < gamma < 3.5) and (r_squared > 0.6)
        else:
            gamma = None
            r_squared = None
            is_scale_free = False
        
        # Network type classification
        network_types = []
        if is_small_world:
            network_types.append("Small-World")
        if is_scale_free:
            network_types.append("Scale-Free")
        if not is_small_world and not is_scale_free:
            if abs(path_ratio - 1) < 0.3 and abs(clustering_ratio - 1) < 2:
                network_types.append("Random")
            else:
                network_types.append("Complex")
        
        return {
            'network_types': network_types,
            'is_small_world': is_small_world,
            'is_scale_free': is_scale_free,
            'clustering_coefficient': C,
            'random_clustering_coefficient': C_random,
            'clustering_ratio': clustering_ratio,
            'avg_path_length': L,
            'random_avg_path_length': L_random,
            'path_length_ratio': path_ratio,
            'power_law_exponent': gamma,
            'power_law_r_squared': r_squared,
            'small_world_coefficient': (clustering_ratio / path_ratio) if path_ratio > 0 else 0
        }
    
    def _resilience_metrics(self) -> dict:
        """Calculate network resilience metrics"""
        
        # Robustness: algebraic connectivity (second smallest eigenvalue of Laplacian)
        try:
            algebraic_connectivity = nx.algebraic_connectivity(self.G)
        except:
            algebraic_connectivity = 0
        
        # Average neighbor degree
        avg_neighbor_degree = nx.average_neighbor_degree(self.G)
        
        return {
            'algebraic_connectivity': algebraic_connectivity,
            'avg_neighbor_degree': np.mean(list(avg_neighbor_degree.values())),
            'node_connectivity': nx.node_connectivity(self.G),
            'edge_connectivity': nx.edge_connectivity(self.G)
        }
    
    def _efficiency_metrics(self) -> dict:
        """Calculate network efficiency metrics"""
        
        # Global efficiency
        global_efficiency = nx.global_efficiency(self.G)
        
        # Local efficiency
        local_efficiency = nx.local_efficiency(self.G)
        
        return {
            'global_efficiency': global_efficiency,
            'local_efficiency': local_efficiency
        }
    
    def _community_analysis(self) -> dict:
        """Analyze community structure"""
        
        # Modularity-based community detection
        communities = nx.community.greedy_modularity_communities(self.G)
        
        # Calculate modularity
        modularity = nx.community.modularity(self.G, communities)
        
        # Community sizes
        community_sizes = [len(c) for c in communities]
        
        return {
            'num_communities': len(communities),
            'modularity': modularity,
            'avg_community_size': np.mean(community_sizes),
            'max_community_size': max(community_sizes),
            'min_community_size': min(community_sizes)
        }
    
    def _spatial_analysis(self) -> dict:
        """Analyze spatial properties"""
        
        # Get edges with distance
        distances = []
        times = []
        costs = []
        
        for u, v, data in self.G.edges(data=True):
            if 'distance' in data and data['distance'] > 0:
                distances.append(data['distance'])
            if 'time' in data and data['time'] > 0:
                times.append(data['time'])
            if 'cost' in data and data['cost'] > 0:
                costs.append(data['cost'])
        
        return {
            'avg_edge_distance': np.mean(distances) if distances else 0,
            'max_edge_distance': max(distances) if distances else 0,
            'min_edge_distance': min(distances) if distances else 0,
            'avg_edge_time': np.mean(times) if times else 0,
            'avg_edge_cost': np.mean(costs) if costs else 0,
            'total_network_length': sum(distances) if distances else 0
        }
    
    def generate_report(self, output_file: str = 'network_analysis_complete.json'):
        """Generate complete analysis report"""
        
        print("\n" + "="*80)
        print("HYDERABAD TRANSPORT NETWORK - COMPLETE ANALYSIS")
        print("="*80)
        
        analysis = self.analyze_complete_network()
        
        # Print summary
        print("\n📊 BASIC PROPERTIES:")
        for key, value in analysis['basic_properties'].items():
            print(f"  {key}: {value}")
        
        print("\n🔗 CONNECTIVITY:")
        for key, value in analysis['connectivity'].items():
            if not isinstance(value, (list, dict)):
                print(f"  {key}: {value}")
        
        print("\n🎯 CENTRALITY (averages):")
        cent = analysis['centrality']
        print(f"  Avg Degree Centrality: {cent['avg_degree_centrality']:.4f}")
        print(f"  Avg Betweenness Centrality: {cent['avg_betweenness_centrality']:.4f}")
        print(f"  Avg Closeness Centrality: {cent['avg_closeness_centrality']:.4f}")
        
        print("\n🕸️  CLUSTERING:")
        for key, value in analysis['clustering'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\n📏 PATH ANALYSIS:")
        for key, value in analysis['path_analysis'].items():
            if not isinstance(value, (list, dict)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n📈 DEGREE DISTRIBUTION:")
        deg = analysis['degree_distribution']
        print(f"  Average Degree: {deg['avg_degree']:.2f}")
        print(f"  Max Degree: {deg['max_degree']}")
        print(f"  Degree Assortativity: {deg['degree_assortativity']:.4f}")
        if deg.get('power_law_exponent'):
            print(f"  Power-law Exponent (γ): {deg['power_law_exponent']:.2f}")
        if deg.get('power_law_fit_r_squared'):
            print(f"  Power-law Fit R²: {deg['power_law_fit_r_squared']:.4f}")
        
        print("\n🌐 NETWORK TYPE IDENTIFICATION:")
        net_type = analysis['network_type']
        print(f"  Network Type(s): {', '.join(net_type['network_types'])}")
        print(f"  Is Small-World: {net_type['is_small_world']}")
        print(f"  Is Scale-Free: {net_type['is_scale_free']}")
        print(f"  Clustering Ratio (C/C_random): {net_type['clustering_ratio']:.2f}")
        print(f"  Path Length Ratio (L/L_random): {net_type['path_length_ratio']:.2f}")
        if net_type['power_law_exponent']:
            print(f"  Power-law Exponent: {net_type['power_law_exponent']:.2f}")
        
        print("\n🛡️  RESILIENCE:")
        for key, value in analysis['resilience'].items():
            if not isinstance(value, (list, dict)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n⚡ EFFICIENCY:")
        for key, value in analysis['efficiency'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\n👥 COMMUNITY STRUCTURE:")
        for key, value in analysis['community'].items():
            if not isinstance(value, (list, dict)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n📍 SPATIAL PROPERTIES:")
        for key, value in analysis['spatial'].items():
            if not isinstance(value, (list, dict)):
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Save to JSON
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            elif obj is None:
                return None
            return obj
        
        analysis_serializable = convert_to_serializable(analysis)
        
        with open(output_file, 'w') as f:
            json.dump(analysis_serializable, f, indent=2)
        
        print(f"\n✅ Complete analysis saved to: {output_file}")
        print("="*80 + "\n")
        
        return analysis


if __name__ == '__main__':
    # Initialize analyzer
    analyzer = NetworkTypeAnalyzer('nodes.csv', 'edges.csv')
    
    # Generate complete report
    analysis = analyzer.generate_report()
