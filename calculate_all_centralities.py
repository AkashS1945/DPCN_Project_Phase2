"""
Comprehensive Centrality Analysis for Hyderabad Transport Network
Calculates 5 centrality measures with top 10 and bottom 10 rankings
"""

import pandas as pd
import json
from collections import defaultdict, deque
import math

class CentralityAnalyzer:
    def __init__(self, nodes_file, edges_file):
        self.nodes = pd.read_csv(nodes_file)
        self.edges = pd.read_csv(edges_file)
        
        # Build adjacency list
        self.graph = defaultdict(set)
        for _, edge in self.edges.iterrows():
            self.graph[edge['from_id']].add(edge['to_id'])
            self.graph[edge['to_id']].add(edge['from_id'])
        
        self.n = len(self.nodes)
        print(f"Network loaded: {self.n} nodes, {len(self.edges)} edges")
    
    def calculate_degree_centrality(self):
        """
        Degree Centrality: Number of direct connections
        Formula: C_D(v) = deg(v) / (n-1)
        """
        print("\n📊 Calculating Degree Centrality...")
        
        degree_data = []
        for _, node in self.nodes.iterrows():
            node_id = node['node_id']
            degree = len(self.graph[node_id])
            # Normalized degree centrality
            degree_centrality = degree / (self.n - 1) if self.n > 1 else 0
            
            degree_data.append({
                'node_id': node_id,
                'name': node.get('name', node_id),
                'layer': node.get('layer', 'unknown'),
                'degree': degree,
                'degree_centrality': degree_centrality
            })
        
        # Sort by degree
        degree_data.sort(key=lambda x: x['degree'], reverse=True)
        
        return {
            'top_10': degree_data[:10],
            'bottom_10': degree_data[-10:],
            'all': degree_data
        }
    
    def bfs_shortest_paths(self, start):
        """BFS to find shortest paths from start node"""
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            for neighbor in self.graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        return distances
    
    def calculate_betweenness_centrality(self, sample_size=200):
        """
        Betweenness Centrality: Fraction of shortest paths passing through node
        Formula: C_B(v) = Σ(σ_st(v) / σ_st) for all s,t pairs
        where σ_st = number of shortest paths from s to t
        σ_st(v) = number of those paths that pass through v
        
        Uses NetworkX for accurate calculation.
        """
        print("\n📊 Calculating Betweenness Centrality...")
        print(f"   Using NetworkX for full graph calculation...")
        
        # Build NetworkX graph
        import networkx as nx
        G_nx = nx.Graph()
        for _, edge in self.edges.iterrows():
            G_nx.add_edge(edge['from_id'], edge['to_id'])
        
        # Calculate betweenness using NetworkX (normalized)
        betweenness_dict = nx.betweenness_centrality(G_nx, normalized=True)
        
        # Create result list
        betweenness_data = []
        for _, node in self.nodes.iterrows():
            node_id = node['node_id']
            betweenness_data.append({
                'node_id': node_id,
                'name': node.get('name', node_id),
                'layer': node.get('layer', 'unknown'),
                'betweenness': betweenness_dict.get(node_id, 0.0)
            })
        
        betweenness_data.sort(key=lambda x: x['betweenness'], reverse=True)
        
        print(f"   ✅ Top node: {betweenness_data[0]['node_id']} = {betweenness_data[0]['betweenness']:.4f}")
        
        return {
            'top_10': betweenness_data[:10],
            'bottom_10': betweenness_data[-10:],
            'all': betweenness_data
        }
    
    def calculate_closeness_centrality(self, sample_size=200):
        """
        Closeness Centrality: Reciprocal of average distance to all other nodes
        Formula: C_C(v) = (n-1) / Σ d(v,u) for all u
        
        Uses NetworkX for accurate calculation.
        """
        print("\n📊 Calculating Closeness Centrality...")
        print(f"   Using NetworkX for full graph calculation...")
        
        # Build NetworkX graph
        import networkx as nx
        G_nx = nx.Graph()
        for _, edge in self.edges.iterrows():
            G_nx.add_edge(edge['from_id'], edge['to_id'])
        
        # Calculate closeness using NetworkX
        closeness_dict = nx.closeness_centrality(G_nx)
        
        closeness_data = []
        for _, node in self.nodes.iterrows():
            node_id = node['node_id']
            closeness_data.append({
                'node_id': node_id,
                'name': node.get('name', node_id),
                'layer': node.get('layer', 'unknown'),
                'closeness': closeness_dict.get(node_id, 0.0)
            })
        
        closeness_data.sort(key=lambda x: x['closeness'], reverse=True)
        
        return {
            'top_10': closeness_data[:10],
            'bottom_10': closeness_data[-10:],
            'all': closeness_data
        }
    
    def calculate_eigenvector_centrality(self, max_iter=100, tol=1e-6):
        """
        Eigenvector Centrality: Influence based on connections to influential nodes
        Formula: x_v = (1/λ) * Σ x_u for all neighbors u
        Iterative power method to find principal eigenvector
        """
        print("\n📊 Calculating Eigenvector Centrality...")
        print(f"   Power iteration method (max {max_iter} iterations)...")
        
        # Initialize centrality values
        centrality = {node_id: 1.0 for node_id in self.graph.keys()}
        
        for iteration in range(max_iter):
            new_centrality = {}
            norm = 0
            
            # Update each node's centrality
            for node_id in self.graph.keys():
                # Sum of neighbors' centralities
                new_value = sum(centrality[neighbor] for neighbor in self.graph[node_id])
                new_centrality[node_id] = new_value
                norm += new_value ** 2
            
            # Normalize
            norm = math.sqrt(norm)
            if norm > 0:
                for node_id in new_centrality:
                    new_centrality[node_id] /= norm
            
            # Check convergence
            diff = sum(abs(new_centrality[n] - centrality[n]) for n in centrality)
            if diff < tol:
                print(f"   ✓ Converged after {iteration + 1} iterations")
                centrality = new_centrality
                break
            
            centrality = new_centrality
            
            if (iteration + 1) % 20 == 0:
                print(f"   Iteration {iteration + 1}/{max_iter}, diff={diff:.6f}")
        
        # Create result list
        eigenvector_data = []
        for _, node in self.nodes.iterrows():
            node_id = node['node_id']
            eigenvector_data.append({
                'node_id': node_id,
                'name': node.get('name', node_id),
                'layer': node.get('layer', 'unknown'),
                'eigenvector': centrality.get(node_id, 0.0)
            })
        
        eigenvector_data.sort(key=lambda x: x['eigenvector'], reverse=True)
        
        return {
            'top_10': eigenvector_data[:10],
            'bottom_10': eigenvector_data[-10:],
            'all': eigenvector_data
        }
    
    def calculate_clustering_coefficient(self):
        """
        Clustering Coefficient: Fraction of neighbors that are also neighbors of each other
        Formula: C(v) = 2 * T(v) / (k_v * (k_v - 1))
        where T(v) = number of triangles through v
        k_v = degree of v
        """
        print("\n📊 Calculating Node Clustering Coefficients...")
        
        clustering_data = []
        
        for i, node_id in enumerate(self.graph.keys()):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{self.n} nodes processed")
            
            neighbors = list(self.graph[node_id])
            degree = len(neighbors)
            
            if degree < 2:
                clustering = 0.0
            else:
                # Count triangles
                triangles = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if neighbors[j] in self.graph[neighbors[i]]:
                            triangles += 1
                
                # Calculate clustering coefficient
                max_edges = degree * (degree - 1) / 2
                clustering = triangles / max_edges if max_edges > 0 else 0
            
            node_info = self.nodes[self.nodes['node_id'] == node_id].iloc[0]
            clustering_data.append({
                'node_id': node_id,
                'name': node_info.get('name', node_id),
                'layer': node_info.get('layer', 'unknown'),
                'degree': degree,
                'clustering': clustering
            })
        
        clustering_data.sort(key=lambda x: x['clustering'], reverse=True)
        
        # Calculate average
        avg_clustering = sum(c['clustering'] for c in clustering_data) / len(clustering_data) if clustering_data else 0
        
        return {
            'top_10': clustering_data[:10],
            'bottom_10': clustering_data[-10:],
            'all': clustering_data,
            'average': avg_clustering
        }

def main():
    print("="*60)
    print("🔬 COMPREHENSIVE CENTRALITY ANALYSIS")
    print("="*60)
    
    analyzer = CentralityAnalyzer('nodes.csv', 'edges.csv')
    
    results = {
        'degree_centrality': analyzer.calculate_degree_centrality(),
        'betweenness_centrality': analyzer.calculate_betweenness_centrality(sample_size=200),
        'closeness_centrality': analyzer.calculate_closeness_centrality(sample_size=200),
        'eigenvector_centrality': analyzer.calculate_eigenvector_centrality(max_iter=100),
        'clustering_coefficient': analyzer.calculate_clustering_coefficient()
    }
    
    # Save results
    output_file = 'centrality_analysis_complete.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for measure, data in results.items():
        print(f"\n{measure.upper().replace('_', ' ')}:")
        if 'top_10' in data:
            print(f"  Top Node: {data['top_10'][0]['name']}")
            if measure == 'degree_centrality':
                print(f"  Value: {data['top_10'][0]['degree']} connections ({data['top_10'][0]['degree_centrality']:.4f} normalized)")
            elif measure == 'clustering_coefficient':
                print(f"  Value: {data['top_10'][0]['clustering']:.4f}")
                print(f"  Average Clustering: {data['average']:.4f}")
            else:
                metric_key = measure.replace('_centrality', '')
                print(f"  Value: {data['top_10'][0][metric_key]:.6f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
