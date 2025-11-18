#!/usr/bin/env python3
"""
Generate comprehensive network analysis for the website.
Creates JSON file with all statistics and metrics.
Includes complex network theory metrics and analysis.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter, deque
import json
from math import radians, sin, cos, sqrt, atan2
import random
from scale_free_analysis import analyze_scale_free
from comprehensive_network_classifier import analyze_network_type


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def analyze_network():
    """Generate comprehensive network analysis."""
    print("📊 Analyzing Hyderabad Multimodal Transport Network...\n")
    
    # Load data
    nodes = pd.read_csv('nodes.csv')
    edges = pd.read_csv('edges.csv')
    regions = pd.read_csv('regions.csv')
    
    analysis = {}
    
    # ===== BASIC STATISTICS =====
    print("1. Basic Statistics...")
    analysis['basic'] = {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'nodes_by_layer': {
            'metro': len(nodes[nodes['layer'] == 'metro']),
            'mmts': len(nodes[nodes['layer'] == 'mmts']),
            'bus': len(nodes[nodes['layer'] == 'bus']),
            'auto': len(nodes[nodes['layer'] == 'auto'])
        },
        'edges_by_mode': {
            'metro': len(edges[edges['mode'] == 'metro']),
            'mmts': len(edges[edges['mode'] == 'mmts']),
            'bus': len(edges[edges['mode'] == 'bus']),
            'auto': len(edges[edges['mode'] == 'auto']),
            'transfer': len(edges[edges['mode'] == 'transfer'])
        }
    }
    
    # ===== CONNECTIVITY ANALYSIS =====
    print("2. Connectivity Analysis...")
    connectivity = {}
    for layer in ['metro', 'mmts', 'bus', 'auto']:
        layer_nodes = nodes[nodes['layer'] == layer]['node_id'].tolist()
        degrees = []
        for node in layer_nodes:
            degree = len(edges[(edges['from_id'] == node) | (edges['to_id'] == node)])
            degrees.append(degree)
        
        connectivity[layer] = {
            'min': int(min(degrees)) if degrees else 0,
            'max': int(max(degrees)) if degrees else 0,
            'avg': round(np.mean(degrees), 2) if degrees else 0,
            'median': int(np.median(degrees)) if degrees else 0
        }
    analysis['connectivity'] = connectivity
    
    # ===== REGIONAL DISTRIBUTION =====
    print("3. Regional Distribution...")
    regional_dist = {}
    for region_id in regions['id']:
        regional_dist[region_id] = {
            'metro': len(nodes[(nodes['layer'] == 'metro') & (nodes['region'] == region_id)]),
            'mmts': len(nodes[(nodes['layer'] == 'mmts') & (nodes['region'] == region_id)]),
            'bus': len(nodes[(nodes['layer'] == 'bus') & (nodes['region'] == region_id)]),
            'auto': len(nodes[(nodes['layer'] == 'auto') & (nodes['region'] == region_id)]),
            'total': len(nodes[nodes['region'] == region_id])
        }
    analysis['regional_distribution'] = regional_dist
    
    # ===== TRANSFER ANALYSIS =====
    print("4. Transfer Analysis...")
    transfers = edges[edges['intra_or_inter'] == 'inter']
    transfer_types = defaultdict(int)
    
    for _, edge in transfers.iterrows():
        from_layer = nodes[nodes['node_id'] == edge['from_id']]['layer'].values
        to_layer = nodes[nodes['node_id'] == edge['to_id']]['layer'].values
        
        if len(from_layer) > 0 and len(to_layer) > 0:
            key = '_'.join(sorted([from_layer[0], to_layer[0]]))
            transfer_types[key] += 1
    
    analysis['transfers'] = {
        'total': len(transfers),
        'percentage': round(len(transfers) * 100 / len(edges), 1),
        'by_type': dict(transfer_types),
        'avg_distance': round(transfers['distance_km'].mean(), 2),
        'avg_time': round(transfers['time_min'].mean(), 2)
    }
    
    # ===== DISTANCE ANALYSIS =====
    print("5. Distance Analysis...")
    distance_stats = {}
    for mode in ['metro', 'mmts', 'bus', 'auto']:
        mode_edges = edges[(edges['mode'] == mode) & (edges['intra_or_inter'] == 'intra')]
        if len(mode_edges) > 0:
            distance_stats[mode] = {
                'min': round(mode_edges['distance_km'].min(), 2),
                'max': round(mode_edges['distance_km'].max(), 2),
                'avg': round(mode_edges['distance_km'].mean(), 2),
                'total': round(mode_edges['distance_km'].sum(), 2)
            }
    analysis['distances'] = distance_stats
    
    # ===== TRAVEL TIME ANALYSIS =====
    print("5b. Travel Time Analysis...")
    time_stats = {}
    for mode in ['metro', 'mmts', 'bus', 'auto']:
        mode_edges = edges[(edges['mode'] == mode) & (edges['intra_or_inter'] == 'intra')]
        if len(mode_edges) > 0:
            time_stats[mode] = {
                'min': round(mode_edges['time_min'].min(), 2),
                'max': round(mode_edges['time_min'].max(), 2),
                'avg': round(mode_edges['time_min'].mean(), 2)
            }
    analysis['travel_times'] = time_stats
    
    # ===== METRO LINE ANALYSIS =====
    print("6. Metro Line Analysis...")
    metro_analysis = {}
    metro_nodes = nodes[nodes['layer'] == 'metro']
    
    for line in ['Red', 'Blue', 'Green']:
        line_nodes = metro_nodes[metro_nodes['metro_line'].str.contains(line, na=False)]
        line_node_ids = line_nodes['node_id'].tolist()
        
        # Count edges for this line
        line_edges = edges[
            (edges['from_id'].isin(line_node_ids)) & 
            (edges['to_id'].isin(line_node_ids)) &
            (edges['mode'] == 'metro')
        ]
        
        metro_analysis[line] = {
            'stations': len(line_nodes),
            'edges': len(line_edges),
            'length_km': round(line_edges['distance_km'].sum() / 2, 2) if len(line_edges) > 0 else 0  # Divide by 2 for bidirectional
        }
    
    # Interchanges
    interchanges = metro_nodes[metro_nodes['metro_line'].str.contains(r'\+', regex=True, na=False)]
    metro_analysis['interchanges'] = {
        'count': len(interchanges),
        'stations': interchanges[['node_id', 'name', 'metro_line']].to_dict('records')
    }
    
    analysis['metro_lines'] = metro_analysis
    
    # ===== ACCESSIBILITY METRICS =====
    print("7. Accessibility Metrics...")
    accessibility = {}
    
    # Separate nodes by layer
    metro_nodes = nodes[nodes['layer'] == 'metro']
    mmts_nodes = nodes[nodes['layer'] == 'mmts']
    rail_nodes = nodes[nodes['layer'].isin(['metro', 'mmts'])]
    bus_nodes = nodes[nodes['layer'] == 'bus']
    auto_nodes = nodes[nodes['layer'] == 'auto']
    
    # Count bus stops near metro (< 500m)
    bus_near_metro = 0
    for _, bus in bus_nodes.iterrows():
        for _, metro in metro_nodes.iterrows():
            dist = haversine(bus['lat'], bus['lon'], metro['lat'], metro['lon'])
            if dist < 0.5:
                bus_near_metro += 1
                break
    
    # Count bus stops near MMTS (< 500m)
    bus_near_mmts = 0
    for _, bus in bus_nodes.iterrows():
        for _, mmts in mmts_nodes.iterrows():
            dist = haversine(bus['lat'], bus['lon'], mmts['lat'], mmts['lon'])
            if dist < 0.5:
                bus_near_mmts += 1
                break
    
    # Count bus stops near any rail (< 500m)
    bus_near_rail = 0
    for _, bus in bus_nodes.iterrows():
        for _, rail in rail_nodes.iterrows():
            dist = haversine(bus['lat'], bus['lon'], rail['lat'], rail['lon'])
            if dist < 0.5:
                bus_near_rail += 1
                break
    
    # Count auto stands near metro (< 500m)
    auto_near_metro = 0
    for _, auto in auto_nodes.iterrows():
        for _, metro in metro_nodes.iterrows():
            dist = haversine(auto['lat'], auto['lon'], metro['lat'], metro['lon'])
            if dist < 0.5:
                auto_near_metro += 1
                break
    
    # Count auto stands near MMTS (< 500m)
    auto_near_mmts = 0
    for _, auto in auto_nodes.iterrows():
        for _, mmts in mmts_nodes.iterrows():
            dist = haversine(auto['lat'], auto['lon'], mmts['lat'], mmts['lon'])
            if dist < 0.5:
                auto_near_mmts += 1
                break
    
    # Count auto stands near any rail (< 500m)
    auto_near_rail = 0
    for _, auto in auto_nodes.iterrows():
        for _, rail in rail_nodes.iterrows():
            dist = haversine(auto['lat'], auto['lon'], rail['lat'], rail['lon'])
            if dist < 0.5:
                auto_near_rail += 1
                break
    
    # Store all metrics
    accessibility['bus_near_metro'] = bus_near_metro
    accessibility['bus_near_mmts'] = bus_near_mmts
    accessibility['bus_near_rail'] = bus_near_rail
    accessibility['bus_far_from_rail'] = len(bus_nodes) - bus_near_rail
    accessibility['bus_near_metro_pct'] = round(bus_near_metro * 100 / len(bus_nodes), 1)
    accessibility['bus_near_mmts_pct'] = round(bus_near_mmts * 100 / len(bus_nodes), 1)
    accessibility['bus_near_rail_pct'] = round(bus_near_rail * 100 / len(bus_nodes), 1)
    accessibility['bus_far_from_rail_pct'] = round((len(bus_nodes) - bus_near_rail) * 100 / len(bus_nodes), 1)
    
    accessibility['auto_near_metro'] = auto_near_metro
    accessibility['auto_near_mmts'] = auto_near_mmts
    accessibility['auto_near_rail'] = auto_near_rail
    accessibility['auto_far_from_rail'] = len(auto_nodes) - auto_near_rail
    accessibility['auto_near_metro_pct'] = round(auto_near_metro * 100 / len(auto_nodes), 1)
    accessibility['auto_near_mmts_pct'] = round(auto_near_mmts * 100 / len(auto_nodes), 1)
    accessibility['auto_near_rail_pct'] = round(auto_near_rail * 100 / len(auto_nodes), 1)
    accessibility['auto_far_from_rail_pct'] = round((len(auto_nodes) - auto_near_rail) * 100 / len(auto_nodes), 1)
    
    accessibility['metro_stations'] = len(metro_nodes)
    accessibility['mmts_stations'] = len(mmts_nodes)
    accessibility['rail_stations'] = len(rail_nodes)
    accessibility['total_bus'] = len(bus_nodes)
    accessibility['total_auto'] = len(auto_nodes)
    accessibility['coverage_area_km2'] = round((nodes['lat'].max() - nodes['lat'].min()) * 111 * 
                                                 (nodes['lon'].max() - nodes['lon'].min()) * 111 * 
                                                 cos(radians(nodes['lat'].mean())), 1)
    
    analysis['accessibility'] = accessibility
    
    # ===== NETWORK EFFICIENCY =====
    print("8. Network Efficiency Metrics...")
    efficiency = {}
    
    # Average degree
    total_degree = sum([len(edges[(edges['from_id'] == node) | (edges['to_id'] == node)]) 
                        for node in nodes['node_id']])
    efficiency['avg_degree'] = round(total_degree / len(nodes), 2)
    
    # Network density (actual edges / possible edges)
    possible_edges = len(nodes) * (len(nodes) - 1) / 2
    efficiency['density'] = round(len(edges) / possible_edges * 100, 4)
    
    # Clustering by layer
    efficiency['avg_distance_km'] = round(edges['distance_km'].mean(), 2)
    efficiency['avg_travel_time_min'] = round(edges['time_min'].mean(), 2)
    efficiency['avg_cost_rs'] = round(edges['cost_rs'].mean(), 2)
    
    analysis['efficiency'] = efficiency
    
    # ===== TOP CONNECTED NODES =====
    print("9. All Connected Nodes (sorted by degree)...")
    node_degrees = []
    for _, node in nodes.iterrows():
        degree = len(edges[(edges['from_id'] == node['node_id']) | (edges['to_id'] == node['node_id'])])
        node_degrees.append({
            'id': node['node_id'],
            'name': node['name'],
            'layer': node['layer'],
            'region': node['region'],
            'degree': degree
        })
    
    # Store ALL nodes sorted by degree (descending) so frontend can slice as needed
    all_nodes_sorted = sorted(node_degrees, key=lambda x: x['degree'], reverse=True)
    analysis['all_connected_nodes'] = all_nodes_sorted
    
    # Also keep top 20 for backward compatibility
    analysis['top_connected_nodes'] = all_nodes_sorted[:20]
    
    # ===== COMPLEX NETWORK METRICS =====
    print("10. Complex Network Analysis...")
    complex_metrics = analyze_complex_network(nodes, edges)
    analysis['complex_networks'] = complex_metrics
    
    # Save to JSON
    print("\n💾 Saving analysis to analysis_data.json...")
    with open('analysis_data.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n✅ Analysis Complete!")
    print(f"   Total Nodes: {analysis['basic']['total_nodes']}")
    print(f"   Total Edges: {analysis['basic']['total_edges']}")
    print(f"   Transfer %: {analysis['transfers']['percentage']}%")
    print(f"   Avg Degree: {analysis['efficiency']['avg_degree']}")
    print(f"   Clustering Coefficient: {analysis['complex_networks']['clustering_coefficient']}")
    print(f"   Avg Path Length: {analysis['complex_networks']['avg_shortest_path']}")
    
    return analysis


def analyze_complex_network(nodes, edges):
    """Analyze complex network properties."""
    
    # Build adjacency list
    adj = defaultdict(set)
    for _, edge in edges.iterrows():
        adj[edge['from_id']].add(edge['to_id'])
        adj[edge['to_id']].add(edge['from_id'])
    
    n = len(nodes)
    node_ids = nodes['node_id'].tolist()
    
    # ===== SCALE-FREE ANALYSIS (Comprehensive Degree Distribution) =====
    # This replaces old basic degree distribution with rigorous power-law analysis
    scale_free_results = analyze_scale_free(nodes_df=nodes, edges_df=edges)
    
    # Extract basic degree statistics from scale-free analysis
    degrees = [len(adj[nid]) for nid in node_ids]
    
    # ===== CLUSTERING COEFFICIENT =====
    def local_clustering(node):
        neighbors = list(adj[node])
        if len(neighbors) < 2:
            return 0
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        actual_edges = sum(1 for i, n1 in enumerate(neighbors) 
                          for n2 in neighbors[i+1:] if n2 in adj[n1])
        return actual_edges / possible_edges if possible_edges > 0 else 0
    
    clustering_coeffs = [local_clustering(nid) for nid in node_ids]
    avg_clustering = np.mean(clustering_coeffs)
    
    # Clustering by layer
    clustering_by_layer = {}
    for layer in ['metro', 'mmts', 'bus', 'auto']:
        layer_nodes = nodes[nodes['layer'] == layer]['node_id'].tolist()
        layer_clustering = [local_clustering(nid) for nid in layer_nodes]
        clustering_by_layer[layer] = round(np.mean(layer_clustering), 4)
    
    # ===== SHORTEST PATH LENGTH (sample-based for large networks) =====
    def bfs_shortest_path(start, end):
        if start == end:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None
    
    # Sample 500 random pairs for path length
    sample_size = min(500, n * (n - 1) // 2)
    path_lengths = []
    sampled = 0
    while sampled < sample_size:
        n1, n2 = random.sample(node_ids, 2)
        path_len = bfs_shortest_path(n1, n2)
        if path_len is not None:
            path_lengths.append(path_len)
            sampled += 1
    
    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    
    # Path length distribution
    path_length_counts = Counter(path_lengths)
    path_length_dist = [{'length': k, 'count': v, 'probability': v/len(path_lengths)} 
                        for k, v in sorted(path_length_counts.items())]
    
    # ===== CENTRALITY MEASURES (using NetworkX for accuracy) =====
    print("   Calculating centrality measures...")
    import networkx as nx
    G_nx = nx.Graph()
    for _, edge in edges.iterrows():
        G_nx.add_edge(edge['from_id'], edge['to_id'])
    
    # Betweenness centrality (normalized)
    betweenness_dict = nx.betweenness_centrality(G_nx, normalized=True)
    top_betweenness = sorted(
        [{'id': nid, 'name': nodes[nodes['node_id']==nid]['name'].values[0],
          'layer': nodes[nodes['node_id']==nid]['layer'].values[0],
          'betweenness': round(betweenness_dict[nid], 4)}
         for nid in betweenness_dict],
        key=lambda x: x['betweenness'], reverse=True
    )[:20]
    
    # Closeness centrality (normalized)
    closeness_dict = nx.closeness_centrality(G_nx)
    top_closeness = sorted(
        [{'id': nid, 'name': nodes[nodes['node_id']==nid]['name'].values[0],
          'layer': nodes[nodes['node_id']==nid]['layer'].values[0],
          'closeness': round(closeness_dict[nid], 4)}
         for nid in closeness_dict],
        key=lambda x: x['closeness'], reverse=True
    )[:20]
    
    # ===== ASSORTATIVITY (degree correlation) =====
    # Pearson correlation of degrees at edge endpoints
    edge_degrees = []
    for _, edge in edges.iterrows():
        d1 = len(adj[edge['from_id']])
        d2 = len(adj[edge['to_id']])
        edge_degrees.append((d1, d2))
    
    if edge_degrees:
        d1s, d2s = zip(*edge_degrees)
        assortativity = np.corrcoef(d1s, d2s)[0, 1]
    else:
        assortativity = 0
    
    # ===== NETWORK RESILIENCE =====
    # Check connectivity after removing top nodes
    def is_connected(nodes_to_check):
        if not nodes_to_check:
            return False
        start = nodes_to_check[0]
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor in nodes_to_check and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(nodes_to_check)
    
    # Original connectivity
    original_connected = is_connected(node_ids)
    
    # Remove top 5% of nodes by degree
    removal_count = max(1, int(0.05 * n))
    top_degree_nodes = sorted(node_ids, key=lambda x: len(adj[x]), reverse=True)[:removal_count]
    remaining_nodes = [nid for nid in node_ids if nid not in top_degree_nodes]
    after_attack_connected = is_connected(remaining_nodes)
    
    # Remove random 5% of nodes
    random_removed = random.sample(node_ids, removal_count)
    remaining_random = [nid for nid in node_ids if nid not in random_removed]
    after_random_connected = is_connected(remaining_random)
    
    # ===== COMPREHENSIVE NETWORK CLASSIFICATION =====
    # Test all network types: Scale-Free, Random, Small-World, Regular, Complete
    print("🔬 Running comprehensive network classification...")
    classification_results = analyze_network_type(nodes_df=nodes, edges_df=edges)
    
    return {
        # Scale-Free & Degree Distribution (comprehensive power-law analysis)
        'scale_free': {
            'is_scale_free': scale_free_results['is_scale_free'],
            'gamma': scale_free_results['gamma'],
            'gamma_loglog': scale_free_results['gamma_loglog'],
            'r_squared': scale_free_results['r_squared'],
            'r_squared_loglog': scale_free_results['r_squared_loglog'],
            'ks_statistic': scale_free_results['ks_statistic'],
            'criteria_passed': scale_free_results['criteria_passed'],
            'criteria_total': scale_free_results['criteria_total']
        },
        'degree_distribution': scale_free_results['degree_distribution'],
        'avg_degree': round(np.mean(degrees), 2),
        'max_degree': int(max(degrees)),
        'min_degree': int(min(degrees)),
        'clustering_coefficient': round(avg_clustering, 4),
        'clustering_by_layer': clustering_by_layer,
        'avg_shortest_path': round(avg_path_length, 2),
        'diameter_estimate': int(max(path_lengths)) if path_lengths else 0,
        'path_length_distribution': path_length_dist,
        'top_betweenness_centrality': top_betweenness,
        'top_closeness_centrality': top_closeness,
        'assortativity_coefficient': round(assortativity, 4),
        'resilience': {
            'originally_connected': original_connected,
            'after_targeted_attack': after_attack_connected,
            'after_random_failure': after_random_connected,
            'nodes_removed_percent': 5,
            'vulnerability_score': 0 if after_attack_connected else 1
        },
        'small_world_coefficient': round(avg_clustering / (avg_path_length if avg_path_length > 0 else 1), 4),
        # Comprehensive Network Classification
        'network_classification': {
            'primary_type': classification_results['classification']['primary_type'],
            'all_types': classification_results['classification']['all_types'],
            'is_random': classification_results['random']['is_random'],
            'is_small_world': classification_results['small_world']['is_small_world'],
            'is_regular': classification_results['regular']['is_regular'],
            'is_complete': classification_results['complete']['is_complete'],
            'sigma': classification_results['small_world']['sigma'],
            'random_test': classification_results['random'],
            'small_world_test': classification_results['small_world']
        }
    }


if __name__ == '__main__':
    analyze_network()
