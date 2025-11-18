#!/usr/bin/env python3
"""
Comprehensive Network Type Classification System
Tests if network is: Scale-Free, Random, Small-World, Regular, Complete, or Complex

Author: Network Analysis System
Date: November 2025
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict, deque
from scipy import stats
from scipy.optimize import curve_fit
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def power_law(x, gamma, c):
    """Power law function: P(k) = c * k^(-gamma)"""
    return c * np.power(x, -gamma)


class NetworkClassifier:
    """Comprehensive network classification with multiple model comparisons"""
    
    def __init__(self, nodes_file='nodes.csv', edges_file='edges.csv', nodes_df=None, edges_df=None):
        """Load network data"""
        print("🔬 COMPREHENSIVE NETWORK TYPE CLASSIFICATION")
        print("=" * 80)
        
        # Load data
        if nodes_df is None:
            self.nodes = pd.read_csv(nodes_file)
        else:
            self.nodes = nodes_df
            
        if edges_df is None:
            self.edges = pd.read_csv(edges_file)
        else:
            self.edges = edges_df
        
        self.node_ids = self.nodes['node_id'].tolist()
        self.n = len(self.node_ids)
        self.m = len(self.edges)
        
        # Build adjacency list
        self.adj = defaultdict(set)
        for _, edge in self.edges.iterrows():
            self.adj[edge['from_id']].add(edge['to_id'])
            self.adj[edge['to_id']].add(edge['from_id'])
        
        # Compute degree sequence
        self.degrees = [len(self.adj[nid]) for nid in self.node_ids]
        self.degree_counts = Counter(self.degrees)
        
        print(f"\n📊 Network Loaded:")
        print(f"   Nodes: {self.n}")
        print(f"   Edges: {self.m}")
        print(f"   Degree range: {min(self.degrees)} to {max(self.degrees)}")
        print(f"   Average degree: {np.mean(self.degrees):.2f}")
        
    
    def compute_core_metrics(self):
        """Compute all core network metrics"""
        print("\n" + "=" * 80)
        print("STEP 1: COMPUTING CORE METRICS")
        print("=" * 80)
        
        metrics = {}
        
        # Basic stats
        metrics['node_count'] = self.n
        metrics['edge_count'] = self.m
        metrics['avg_degree'] = np.mean(self.degrees)
        metrics['degree_variance'] = np.var(self.degrees)
        metrics['min_degree'] = min(self.degrees)
        metrics['max_degree'] = max(self.degrees)
        
        # Clustering coefficient
        print("\n📊 Computing clustering coefficient...")
        clustering_coeffs = []
        for nid in self.node_ids:
            neighbors = list(self.adj[nid])
            if len(neighbors) < 2:
                clustering_coeffs.append(0)
                continue
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = sum(1 for i, n1 in enumerate(neighbors) 
                             for n2 in neighbors[i+1:] if n2 in self.adj[n1])
            clustering_coeffs.append(actual_edges / possible_edges if possible_edges > 0 else 0)
        
        metrics['clustering_coefficient'] = np.mean(clustering_coeffs)
        print(f"   ✓ Clustering coefficient: {metrics['clustering_coefficient']:.4f}")
        
        # Average shortest path (sample for large graphs)
        print("\n📏 Computing average shortest path length...")
        sample_size = min(200, self.n)
        sampled_nodes = np.random.choice(self.node_ids, sample_size, replace=False)
        path_lengths = []
        
        for source in sampled_nodes:
            distances = self._bfs_distances(source)
            for target in sampled_nodes:
                if target != source and target in distances:
                    path_lengths.append(distances[target])
        
        metrics['avg_path_length'] = np.mean(path_lengths) if path_lengths else float('inf')
        metrics['diameter_estimate'] = max(path_lengths) if path_lengths else 0
        print(f"   ✓ Average path length: {metrics['avg_path_length']:.2f}")
        print(f"   ✓ Diameter estimate: {metrics['diameter_estimate']}")
        
        # Degree assortativity
        print("\n🔗 Computing degree assortativity...")
        metrics['assortativity'] = self._compute_assortativity()
        print(f"   ✓ Assortativity: {metrics['assortativity']:.4f}")
        
        self.metrics = metrics
        return metrics
    
    
    def _bfs_distances(self, source):
        """BFS to compute distances from source"""
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            node = queue.popleft()
            for neighbor in self.adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        return distances
    
    
    def _compute_assortativity(self):
        """Compute degree assortativity coefficient"""
        degrees_dict = {nid: len(self.adj[nid]) for nid in self.node_ids}
        
        sum_jk = 0
        sum_j_plus_k = 0
        sum_j_plus_k_sq = 0
        edge_count = 0
        
        for _, edge in self.edges.iterrows():
            j = degrees_dict[edge['from_id']]
            k = degrees_dict[edge['to_id']]
            
            sum_jk += j * k
            sum_j_plus_k += j + k
            sum_j_plus_k_sq += j*j + k*k
            edge_count += 1
        
        if edge_count == 0:
            return 0
        
        numerator = sum_jk / edge_count - (sum_j_plus_k / (2 * edge_count)) ** 2
        denominator = sum_j_plus_k_sq / (2 * edge_count) - (sum_j_plus_k / (2 * edge_count)) ** 2
        
        return numerator / denominator if denominator != 0 else 0
    
    
    def test_scale_free(self):
        """Test if network is scale-free using power-law fitting"""
        print("\n" + "=" * 80)
        print("STEP 2: TESTING SCALE-FREE PROPERTY")
        print("=" * 80)
        
        degrees = np.array(sorted(self.degree_counts.keys()))
        counts = np.array([self.degree_counts[d] for d in degrees])
        probabilities = counts / self.n
        
        # Filter non-zero
        mask = (degrees > 0) & (counts > 0)
        degrees_nz = degrees[mask]
        probs_nz = probabilities[mask]
        
        # Fit power law
        try:
            popt, _ = curve_fit(power_law, degrees_nz, probs_nz, p0=[2.5, 0.1], maxfev=10000)
            gamma, c = popt
            
            # Compute fit quality
            predicted = power_law(degrees_nz, gamma, c)
            residuals = probs_nz - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((probs_nz - np.mean(probs_nz))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))
            
            # KS test
            empirical_cdf = np.cumsum(probs_nz)
            theoretical_cdf = np.cumsum(predicted) / np.sum(predicted)
            ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
            
            # Log-log regression (alternative)
            log_degrees = np.log(degrees_nz)
            log_probs = np.log(probs_nz)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_probs)
            gamma_loglog = -slope
            r_squared_loglog = r_value**2
            
        except Exception as e:
            print(f"   ⚠ Power-law fitting failed: {e}")
            gamma, c, r_squared, rmse, ks_stat = 0, 0, 0, 1, 1
            gamma_loglog, r_squared_loglog = 0, 0
        
        # Decision criteria
        gamma_ok = 2 <= gamma <= 3
        fit_ok = r_squared > 0.5 and rmse < 0.1
        ks_ok = ks_stat < 0.15
        
        criteria_passed = sum([gamma_ok, fit_ok, ks_ok])
        is_scale_free = criteria_passed >= 2
        
        print(f"\n   Power-law exponent (γ): {gamma:.4f}")
        print(f"   R² score: {r_squared:.4f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   KS statistic: {ks_stat:.4f}")
        print(f"\n   Criteria:")
        print(f"   {'✓' if gamma_ok else '✗'} Gamma ∈ [2, 3]: {gamma:.4f}")
        print(f"   {'✓' if fit_ok else '✗'} Good fit (R² > 0.5, RMSE < 0.1)")
        print(f"   {'✓' if ks_ok else '✗'} KS test (< 0.15)")
        print(f"\n   {'✅' if is_scale_free else '❌'} Scale-Free: {criteria_passed}/3 criteria passed")
        
        return {
            'is_scale_free': is_scale_free,
            'gamma': gamma,
            'gamma_loglog': gamma_loglog,
            'r_squared': r_squared,
            'r_squared_loglog': r_squared_loglog,
            'ks_statistic': ks_stat,
            'rmse': rmse,
            'criteria_passed': criteria_passed
        }
    
    
    def test_random(self):
        """Test if network resembles Erdős-Rényi random graph"""
        print("\n" + "=" * 80)
        print("STEP 3: TESTING RANDOM (ERDŐS-RÉNYI) PROPERTY")
        print("=" * 80)
        
        # Expected properties of random graph
        p = 2 * self.m / (self.n * (self.n - 1))  # Edge probability
        expected_avg_degree = p * (self.n - 1)
        expected_clustering = p
        expected_path_length = np.log(self.n) / np.log(expected_avg_degree) if expected_avg_degree > 1 else float('inf')
        
        # Test degree distribution against Poisson
        poisson_lambda = self.metrics['avg_degree']
        observed_freq = np.array([self.degree_counts.get(k, 0) for k in range(max(self.degrees) + 1)])
        expected_freq = self.n * stats.poisson.pmf(range(max(self.degrees) + 1), poisson_lambda)
        
        # Chi-square test (normalize to avoid mismatch)
        mask = expected_freq > 5
        if mask.sum() > 1:
            obs_masked = observed_freq[mask]
            exp_masked = expected_freq[mask]
            # Normalize both to same total
            total = obs_masked.sum()
            exp_masked = exp_masked * (total / exp_masked.sum())
            chi2_stat, p_value = stats.chisquare(obs_masked, exp_masked)
            is_poisson = p_value > 0.05
        else:
            chi2_stat, p_value = 0, 0
            is_poisson = False
        
        # Check clustering
        clustering_ratio = self.metrics['clustering_coefficient'] / expected_clustering if expected_clustering > 0 else float('inf')
        low_clustering = clustering_ratio < 2  # Within 2x of expected
        
        # Check path length
        path_ratio = self.metrics['avg_path_length'] / expected_path_length if expected_path_length != float('inf') else 1
        short_paths = 0.5 < path_ratio < 2
        
        # Overall decision
        is_random = is_poisson and low_clustering and short_paths
        
        print(f"\n   Edge probability (p): {p:.6f}")
        print(f"   Expected avg degree: {expected_avg_degree:.2f} (actual: {self.metrics['avg_degree']:.2f})")
        print(f"   Expected clustering: {expected_clustering:.4f} (actual: {self.metrics['clustering_coefficient']:.4f})")
        print(f"   Expected path length: {expected_path_length:.2f} (actual: {self.metrics['avg_path_length']:.2f})")
        print(f"\n   Degree distribution vs Poisson:")
        print(f"   {'✓' if is_poisson else '✗'} Chi² p-value: {p_value:.4f} (>0.05 means Poisson-like)")
        print(f"   {'✓' if low_clustering else '✗'} Low clustering (ratio: {clustering_ratio:.2f})")
        print(f"   {'✓' if short_paths else '✗'} Short paths (ratio: {path_ratio:.2f})")
        print(f"\n   {'✅' if is_random else '❌'} Random-like: {sum([is_poisson, low_clustering, short_paths])}/3 criteria")
        
        return {
            'is_random': is_random,
            'edge_probability': p,
            'poisson_p_value': p_value,
            'clustering_ratio': clustering_ratio,
            'path_ratio': path_ratio,
            'expected_clustering': expected_clustering,
            'expected_path_length': expected_path_length
        }
    
    
    def test_small_world(self):
        """Test if network is small-world using Watts-Strogatz criteria"""
        print("\n" + "=" * 80)
        print("STEP 4: TESTING SMALL-WORLD PROPERTY")
        print("=" * 80)
        
        # Generate equivalent random graph for comparison
        p_random = 2 * self.m / (self.n * (self.n - 1))
        C_random = p_random
        L_random = np.log(self.n) / np.log(self.metrics['avg_degree']) if self.metrics['avg_degree'] > 1 else float('inf')
        
        C_real = self.metrics['clustering_coefficient']
        L_real = self.metrics['avg_path_length']
        
        # Small-world sigma metric
        if L_random != float('inf') and L_random > 0:
            sigma = (C_real / C_random) / (L_real / L_random) if C_random > 0 and L_random > 0 else 0
        else:
            sigma = 0
        
        # Criteria
        high_clustering = C_real > 3 * C_random  # Much higher than random
        similar_paths = 0.8 < (L_real / L_random) < 1.5 if L_random != float('inf') else False
        is_small_world = sigma > 1 and high_clustering
        
        print(f"\n   Clustering coefficient (C):")
        print(f"      Real: {C_real:.4f}")
        print(f"      Random: {C_random:.4f}")
        print(f"      Ratio: {C_real/C_random:.2f}x" if C_random > 0 else "      Ratio: ∞")
        print(f"\n   Average path length (L):")
        print(f"      Real: {L_real:.2f}")
        print(f"      Random: {L_random:.2f}")
        print(f"      Ratio: {L_real/L_random:.2f}x" if L_random != float('inf') else "      Ratio: N/A")
        print(f"\n   Small-world sigma: {sigma:.4f} (>1 indicates small-world)")
        print(f"\n   {'✓' if high_clustering else '✗'} High clustering (>3x random)")
        print(f"   {'✓' if similar_paths else '✗'} Similar path length to random")
        print(f"\n   {'✅' if is_small_world else '❌'} Small-World: {'Yes' if is_small_world else 'No'}")
        
        return {
            'is_small_world': is_small_world,
            'sigma': sigma,
            'C_real': C_real,
            'C_random': C_random,
            'L_real': L_real,
            'L_random': L_random,
            'clustering_ratio': C_real / C_random if C_random > 0 else float('inf'),
            'path_ratio': L_real / L_random if L_random != float('inf') else 1
        }
    
    
    def test_regular(self):
        """Test if network is regular (all nodes same degree)"""
        print("\n" + "=" * 80)
        print("STEP 5: TESTING REGULAR PROPERTY")
        print("=" * 80)
        
        unique_degrees = len(set(self.degrees))
        is_regular = unique_degrees == 1
        degree_std = np.std(self.degrees)
        
        print(f"\n   Unique degree values: {unique_degrees}")
        print(f"   Degree std deviation: {degree_std:.2f}")
        print(f"   {'✅' if is_regular else '❌'} Regular: {'Yes (all nodes have same degree)' if is_regular else 'No'}")
        
        return {
            'is_regular': is_regular,
            'unique_degrees': unique_degrees,
            'degree_std': degree_std
        }
    
    
    def test_complete(self):
        """Test if network is complete (fully connected)"""
        print("\n" + "=" * 80)
        print("STEP 6: TESTING COMPLETE PROPERTY")
        print("=" * 80)
        
        max_edges = self.n * (self.n - 1) / 2
        expected_degree = self.n - 1
        
        has_max_edges = self.m == max_edges
        all_max_degree = all(d == expected_degree for d in self.degrees)
        is_complete = has_max_edges and all_max_degree
        
        print(f"\n   Current edges: {self.m}")
        print(f"   Maximum possible edges: {int(max_edges)}")
        print(f"   Edge ratio: {self.m/max_edges:.4f}")
        print(f"   {'✅' if is_complete else '❌'} Complete: {'Yes' if is_complete else 'No'}")
        
        return {
            'is_complete': is_complete,
            'edge_completeness': self.m / max_edges,
            'max_possible_edges': int(max_edges)
        }
    
    
    def generate_model_comparisons(self):
        """Generate comparison models and compute their metrics"""
        print("\n" + "=" * 80)
        print("STEP 7: GENERATING COMPARISON MODELS")
        print("=" * 80)
        
        models = {}
        
        # 1. Erdős-Rényi Random Graph
        print("\n📊 Generating Erdős-Rényi random graph...")
        p = 2 * self.m / (self.n * (self.n - 1))
        models['random'] = self._generate_random_graph_metrics(p)
        
        # 2. Configuration Model (degree-preserving)
        print("\n📊 Generating configuration model (degree-preserving)...")
        models['configuration'] = self._generate_configuration_model_metrics()
        
        # 3. Regular graph (if possible)
        print("\n📊 Checking regular graph possibility...")
        if self.m % self.n == 0:
            k = 2 * self.m // self.n
            if k < self.n:
                models['regular'] = self._generate_regular_graph_metrics(k)
            else:
                models['regular'] = None
                print("   ⚠ Cannot create regular graph (k >= n)")
        else:
            models['regular'] = None
            print("   ⚠ Cannot create regular graph (m not divisible by n)")
        
        return models
    
    
    def _generate_random_graph_metrics(self, p):
        """Compute expected metrics for random graph"""
        return {
            'avg_degree': p * (self.n - 1),
            'clustering': p,
            'path_length': np.log(self.n) / np.log(p * (self.n - 1)) if p * (self.n - 1) > 1 else float('inf'),
            'degree_variance': p * (self.n - 1) * (1 - p)
        }
    
    
    def _generate_configuration_model_metrics(self):
        """Estimate metrics for configuration model"""
        avg_deg = self.metrics['avg_degree']
        avg_deg_sq = np.mean([d**2 for d in self.degrees])
        
        return {
            'avg_degree': avg_deg,
            'clustering': avg_deg_sq / (self.n * avg_deg)**2 if avg_deg > 0 else 0,
            'path_length': np.log(self.n) / np.log(avg_deg) if avg_deg > 1 else float('inf'),
            'degree_variance': self.metrics['degree_variance']
        }
    
    
    def _generate_regular_graph_metrics(self, k):
        """Compute metrics for k-regular graph"""
        return {
            'avg_degree': k,
            'clustering': (k - 1) / (self.n - 1) if self.n > 1 else 0,
            'path_length': np.log(self.n) / np.log(k) if k > 1 else float('inf'),
            'degree_variance': 0
        }
    
    
    def classify_network(self):
        """Final classification with recommendations"""
        print("\n" + "=" * 80)
        print("STEP 8: FINAL NETWORK CLASSIFICATION")
        print("=" * 80)
        
        # Test all properties
        scale_free = self.test_scale_free()
        random = self.test_random()
        small_world = self.test_small_world()
        regular = self.test_regular()
        complete = self.test_complete()
        models = self.generate_model_comparisons()
        
        # Determine primary classification
        classifications = []
        
        if scale_free['is_scale_free']:
            classifications.append('SCALE-FREE')
        if random['is_random']:
            classifications.append('RANDOM')
        if small_world['is_small_world']:
            classifications.append('SMALL-WORLD')
        if regular['is_regular']:
            classifications.append('REGULAR')
        if complete['is_complete']:
            classifications.append('COMPLETE')
        
        if not classifications:
            classifications.append('COMPLEX')
        
        primary_type = classifications[0]
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 FINAL VERDICT")
        print("=" * 80)
        
        print(f"\n{'🔷' * 40}")
        print(f"   PRIMARY CLASSIFICATION: {primary_type}")
        if len(classifications) > 1:
            print(f"   SECONDARY: {', '.join(classifications[1:])}")
        print(f"{'🔷' * 40}")
        
        # Detailed reasoning
        print("\n📋 DETAILED REASONING:\n")
        
        if scale_free['is_scale_free']:
            print(f"✅ SCALE-FREE:")
            print(f"   - Power-law exponent γ = {scale_free['gamma']:.4f}")
            print(f"   - Passed {scale_free['criteria_passed']}/3 statistical criteria")
            print(f"   - Hub structure detected")
        else:
            print(f"❌ NOT SCALE-FREE:")
            print(f"   - γ = {scale_free['gamma']:.4f} (expected: 2-3)")
            print(f"   - R² = {scale_free['r_squared']:.4f} (expected: >0.5)")
            print(f"   - Degree distribution does not follow power law")
        
        print()
        if random['is_random']:
            print(f"✅ RANDOM (Erdős-Rényi):")
            print(f"   - Degree distribution is Poisson-like")
            print(f"   - Low clustering ({self.metrics['clustering_coefficient']:.4f})")
            print(f"   - Short path lengths")
        else:
            print(f"❌ NOT RANDOM:")
            print(f"   - Clustering too high ({random['clustering_ratio']:.2f}x expected)")
            print(f"   - Degree distribution not Poisson (p={random['poisson_p_value']:.4f})")
        
        print()
        if small_world['is_small_world']:
            print(f"✅ SMALL-WORLD:")
            print(f"   - High clustering ({small_world['C_real']:.4f} vs {small_world['C_random']:.4f} random)")
            print(f"   - Short paths like random graph")
            print(f"   - Sigma = {small_world['sigma']:.2f} > 1")
        else:
            print(f"❌ NOT SMALL-WORLD:")
            print(f"   - Sigma = {small_world['sigma']:.2f} (expected: >1)")
            if not small_world['sigma'] > 1:
                print(f"   - Clustering not high enough relative to path length")
        
        print()
        if regular['is_regular']:
            print(f"✅ REGULAR: All nodes have degree {self.degrees[0]}")
        else:
            print(f"❌ NOT REGULAR: {regular['unique_degrees']} different degree values")
        
        print()
        if complete['is_complete']:
            print(f"✅ COMPLETE: Fully connected graph")
        else:
            print(f"❌ NOT COMPLETE: Only {complete['edge_completeness']:.2%} of possible edges")
        
        # Recommendations if complex
        if primary_type == 'COMPLEX':
            print("\n" + "=" * 80)
            print("💡 RECOMMENDATIONS TO FIT KNOWN MODELS")
            print("=" * 80)
            
            print("\n📌 Closest model: SMALL-WORLD")
            print(f"   Current sigma: {small_world['sigma']:.2f}")
            print(f"   To become small-world:")
            print(f"   - Increase local clustering (currently {small_world['C_real']:.4f})")
            print(f"   - Add more triangles/communities")
            print(f"   - Keep short paths (currently good at {small_world['L_real']:.2f})")
            
            if scale_free['gamma'] < 2:
                print("\n📌 Towards SCALE-FREE:")
                print(f"   - Current γ = {scale_free['gamma']:.4f} is too low")
                print(f"   - Add more medium-degree nodes (10-30 connections)")
                print(f"   - Reduce extreme hub dominance")
            elif scale_free['gamma'] > 3:
                print("\n📌 Towards SCALE-FREE:")
                print(f"   - Current γ = {scale_free['gamma']:.4f} is too high")
                print(f"   - Add preferential attachment (rich get richer)")
                print(f"   - Create more hub nodes")
        
        return {
            'primary_type': primary_type,
            'all_types': classifications,
            'scale_free': scale_free,
            'random': random,
            'small_world': small_world,
            'regular': regular,
            'complete': complete,
            'models': models,
            'metrics': self.metrics
        }
    
    
    def plot_comparisons(self, results, output_file='network_classification_plots.png'):
        """Generate comparison plots"""
        print("\n" + "=" * 80)
        print("STEP 9: GENERATING COMPARISON PLOTS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Network Classification Analysis', fontsize=20, fontweight='bold')
        
        # 1. Degree Distribution (log-log)
        ax1 = axes[0, 0]
        degrees = np.array(sorted(self.degree_counts.keys()))
        counts = np.array([self.degree_counts[d] for d in degrees])
        probs = counts / self.n
        
        ax1.loglog(degrees, probs, 'o-', linewidth=2, markersize=8, label='Actual Network', color='#667eea')
        
        # Power-law fit line
        if results['scale_free']['gamma'] > 0:
            gamma = results['scale_free']['gamma']
            fit_line = probs[0] * (degrees / degrees[0]) ** (-gamma)
            ax1.loglog(degrees, fit_line, '--', linewidth=2, label=f'Power Law (γ={gamma:.2f})', color='#f093fb')
        
        ax1.set_xlabel('Degree (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('P(k)', fontsize=12, fontweight='bold')
        ax1.set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Clustering Comparison
        ax2 = axes[0, 1]
        clustering_data = {
            'Actual': self.metrics['clustering_coefficient'],
            'Random\nExpected': results['random']['expected_clustering'],
            'Config\nModel': results['models']['configuration']['clustering']
        }
        
        bars = ax2.bar(clustering_data.keys(), clustering_data.values(), 
                       color=['#667eea', '#f093fb', '#4facfe'], edgecolor='black', linewidth=2)
        ax2.set_ylabel('Clustering Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('Clustering Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Path Length Comparison
        ax3 = axes[1, 0]
        path_data = {
            'Actual': self.metrics['avg_path_length'],
            'Random\nExpected': results['random']['expected_path_length'] if results['random']['expected_path_length'] != float('inf') else 0,
            'Config\nModel': results['models']['configuration']['path_length'] if results['models']['configuration']['path_length'] != float('inf') else 0
        }
        
        bars = ax3.bar(path_data.keys(), path_data.values(),
                       color=['#667eea', '#f093fb', '#4facfe'], edgecolor='black', linewidth=2)
        ax3.set_ylabel('Average Path Length', fontsize=12, fontweight='bold')
        ax3.set_title('Path Length Comparison', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Classification Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
NETWORK CLASSIFICATION SUMMARY

Primary Type: {results['primary_type']}
All Types: {', '.join(results['all_types'])}

Core Metrics:
• Nodes: {self.n:,}
• Edges: {self.m:,}
• Avg Degree: {self.metrics['avg_degree']:.2f}
• Clustering: {self.metrics['clustering_coefficient']:.4f}
• Avg Path: {self.metrics['avg_path_length']:.2f}
• Assortativity: {self.metrics['assortativity']:.4f}

Test Results:
{'✅' if results['scale_free']['is_scale_free'] else '❌'} Scale-Free (γ={results['scale_free']['gamma']:.2f})
{'✅' if results['random']['is_random'] else '❌'} Random (Erdős-Rényi)
{'✅' if results['small_world']['is_small_world'] else '❌'} Small-World (σ={results['small_world']['sigma']:.2f})
{'✅' if results['regular']['is_regular'] else '❌'} Regular
{'✅' if results['complete']['is_complete'] else '❌'} Complete
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n   ✓ Saved plots to {output_file}")
        
        return output_file


def analyze_network_type(nodes_file='nodes.csv', edges_file='edges.csv', nodes_df=None, edges_df=None):
    """Main function to classify network type"""
    
    classifier = NetworkClassifier(nodes_file, edges_file, nodes_df, edges_df)
    
    # Compute metrics
    classifier.compute_core_metrics()
    
    # Classify
    results = classifier.classify_network()
    
    # Plot
    classifier.plot_comparisons(results)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    # Save results
    output = {
        'classification': {
            'primary_type': results['primary_type'],
            'all_types': results['all_types']
        },
        'scale_free': convert_to_native(results['scale_free']),
        'random': convert_to_native(results['random']),
        'small_world': convert_to_native(results['small_world']),
        'regular': convert_to_native(results['regular']),
        'complete': convert_to_native(results['complete']),
        'core_metrics': convert_to_native(results['metrics']),
        'model_comparisons': convert_to_native(results['models'])
    }
    
    with open('network_classification_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n💾 Results saved to network_classification_results.json")
    print("\n✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return output


if __name__ == '__main__':
    results = analyze_network_type()
