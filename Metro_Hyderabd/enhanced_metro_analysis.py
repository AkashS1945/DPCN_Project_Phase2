#!/usr/bin/env python3
"""
Enhanced Metro Network Analysis with Timing and Importance Analysis
Similar to MMTS analysis
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict

class EnhancedMetroAnalyzer:
    def __init__(self, gtfs_path="../Metro_Hyderabd"):
        self.gtfs_path = Path(gtfs_path)
        self.load_gtfs_data()
        self.G = None
        self.metrics = {}
        self.station_timings = {}
        
    def load_gtfs_data(self):
        """Load all GTFS files"""
        print("Loading GTFS data...")
        self.stops = pd.read_csv(self.gtfs_path / "stops.txt")
        self.routes = pd.read_csv(self.gtfs_path / "routes.txt")
        self.trips = pd.read_csv(self.gtfs_path / "trips.txt")
        self.stop_times = pd.read_csv(self.gtfs_path / "stop_times.txt")
        
        print(f"✓ Loaded {len(self.routes)} routes")
        print(f"✓ Loaded {len(self.stops)} stops")
        print(f"✓ Loaded {len(self.stop_times)} stop times")
    
    def build_network(self):
        """Build network graph from stop_times - using parent stations only"""
        print("\nBuilding network...")
        self.G = nx.Graph()
        
        # Get only parent stations (location_type == 1) for network analysis
        parent_stations = self.stops[self.stops['location_type'] == 1].copy()
        parent_ids = set(parent_stations['stop_id'].tolist())
        
        print(f"  Using {len(parent_stations)} parent stations for network graph")
        
        # Add nodes for parent stations
        for _, stop in parent_stations.iterrows():
            self.G.add_node(
                stop['stop_id'],
                name=stop['stop_name'],
                lat=stop['stop_lat'],
                lon=stop['stop_lon']
            )
        
        # Map platform stops to parent stations
        platform_to_parent = {}
        for _, stop in self.stops.iterrows():
            if stop['location_type'] in [0, 2]:  # Platform or other stop
                # Extract parent ID by removing numeric suffix
                stop_id = str(stop['stop_id'])
                parent_id = stop_id.rstrip('0123456789')
                if parent_id in parent_ids:
                    platform_to_parent[stop['stop_id']] = parent_id
        
        # Add edges from stop_times (consecutive stops in each trip)
        for trip_id in self.stop_times['trip_id'].unique():
            trip_stops = self.stop_times[self.stop_times['trip_id'] == trip_id].sort_values('stop_sequence')
            stops_list = trip_stops['stop_id'].tolist()
            
            for i in range(len(stops_list) - 1):
                # Map to parent stations
                stop1 = platform_to_parent.get(stops_list[i], stops_list[i])
                stop2 = platform_to_parent.get(stops_list[i + 1], stops_list[i + 1])
                
                if stop1 in self.G.nodes and stop2 in self.G.nodes and stop1 != stop2:
                    self.G.add_edge(stop1, stop2)
        
        print(f"✓ Network: {self.G.number_of_nodes()} stations, {self.G.number_of_edges()} connections")
    
    def analyze_station_timings(self):
        """Analyze timing information for each parent station (aggregates all platforms)"""
        print("\nAnalyzing station timings...")
        
        # Get parent stations
        parent_stations = self.stops[self.stops['location_type'] == 1].copy()
        parent_ids = set(parent_stations['stop_id'].tolist())
        
        # Map platform stops to parent stations
        platform_to_parent = {}
        for _, stop in self.stops.iterrows():
            if stop['location_type'] in [0, 2]:  # Platform or other stop
                stop_id = str(stop['stop_id'])
                parent_id = stop_id.rstrip('0123456789')
                if parent_id in parent_ids:
                    platform_to_parent[stop['stop_id']] = parent_id
        
        # Aggregate timing data by parent station
        station_times = defaultdict(list)
        
        for _, row in self.stop_times.iterrows():
            stop_id = row['stop_id']
            parent_id = platform_to_parent.get(stop_id, stop_id)
            
            if parent_id in parent_ids:
                try:
                    arr_time = row['arrival_time']
                    arr_parts = str(arr_time).split(':')
                    arr_minutes = int(arr_parts[0]) * 60 + int(arr_parts[1])
                    station_times[parent_id].append(arr_minutes)
                except:
                    pass
        
        # Calculate statistics for each parent station
        for parent_id, arrival_times in station_times.items():
            if len(arrival_times) > 0:
                stop_name = self.stops[self.stops['stop_id'] == parent_id].iloc[0]['stop_name']
                
                # Calculate statistics
                total_services = len(arrival_times)
                first_service = min(arrival_times)
                last_service = max(arrival_times)
                
                # Calculate average frequency
                sorted_times = sorted(arrival_times)
                if len(sorted_times) > 1:
                    gaps = [sorted_times[i+1] - sorted_times[i] for i in range(len(sorted_times)-1)]
                    avg_frequency = np.mean(gaps) if gaps else 0
                else:
                    avg_frequency = 0
                
                # Identify peak hours
                morning_services = sum(1 for t in arrival_times if 360 <= t <= 600)  # 6-10 AM
                evening_services = sum(1 for t in arrival_times if 1020 <= t <= 1260)  # 5-9 PM
                
                self.station_timings[parent_id] = {
                    'stop_name': stop_name,
                    'total_services': total_services,
                    'first_service': self._minutes_to_time(first_service),
                    'last_service': self._minutes_to_time(last_service),
                    'avg_frequency_minutes': round(avg_frequency, 1),
                    'peak_hours': f"Morning: {morning_services} services, Evening: {evening_services} services"
                }
        
        print(f"✓ Analyzed {len(self.station_timings)} stations")
    
    def _minutes_to_time(self, minutes):
        """Convert minutes since midnight to HH:MM format"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def compute_centrality(self):
        """Compute all centrality measures"""
        print("\nComputing centrality measures...")
        
        self.degree_cent = nx.degree_centrality(self.G)
        self.between_cent = nx.betweenness_centrality(self.G)
        self.close_cent = nx.closeness_centrality(self.G)
        
        try:
            self.eigen_cent = nx.eigenvector_centrality(self.G, max_iter=1000)
        except:
            self.eigen_cent = {node: 0 for node in self.G.nodes()}
        
        print("✓ Centrality computation complete")
    
    def get_top_critical_hubs(self, n=10):
        """Get top N critical hubs combining timing and centrality data - deduplicated by station name"""
        hub_dict = {}
        
        for stop_id in self.station_timings.keys():
            if stop_id in self.between_cent:
                timing = self.station_timings[stop_id]
                station_name = timing['stop_name']
                
                # If we've seen this station before, only keep the one with higher betweenness
                if station_name in hub_dict:
                    if self.between_cent[stop_id] > hub_dict[station_name]['betweenness_centrality']:
                        hub_dict[station_name] = {
                            'stop_id': stop_id,
                            'name': station_name,
                            'betweenness_centrality': self.between_cent[stop_id],
                            'degree_centrality': self.degree_cent[stop_id],
                            'total_services': timing['total_services'],
                            'first_service': timing['first_service'],
                            'last_service': timing['last_service'],
                            'avg_frequency_minutes': timing['avg_frequency_minutes'],
                            'peak_hours': timing['peak_hours']
                        }
                else:
                    hub_dict[station_name] = {
                        'stop_id': stop_id,
                        'name': station_name,
                        'betweenness_centrality': self.between_cent[stop_id],
                        'degree_centrality': self.degree_cent[stop_id],
                        'total_services': timing['total_services'],
                        'first_service': timing['first_service'],
                        'last_service': timing['last_service'],
                        'avg_frequency_minutes': timing['avg_frequency_minutes'],
                        'peak_hours': timing['peak_hours']
                    }
        
        # Convert to list and round values
        hub_data = []
        for data in hub_dict.values():
            data['betweenness_centrality'] = round(data['betweenness_centrality'], 4)
            data['degree_centrality'] = round(data['degree_centrality'], 4)
            hub_data.append(data)
        
        # Sort by betweenness centrality (descending)
        hub_data.sort(key=lambda x: x['betweenness_centrality'], reverse=True)
        return hub_data[:n]
    
    def get_busiest_stations(self, n=5):
        """Get top N busiest stations by service frequency - deduplicated by station name"""
        station_dict = {}
        
        for stop_id, timing in self.station_timings.items():
            station_name = timing['stop_name']
            
            # If duplicate, keep the one with more services
            if station_name in station_dict:
                if timing['total_services'] > station_dict[station_name]['total_services']:
                    station_dict[station_name] = {
                        'name': station_name,
                        'total_services': timing['total_services'],
                        'first_service': timing['first_service'],
                        'last_service': timing['last_service'],
                        'avg_frequency_minutes': timing['avg_frequency_minutes']
                    }
            else:
                station_dict[station_name] = {
                    'name': station_name,
                    'total_services': timing['total_services'],
                    'first_service': timing['first_service'],
                    'last_service': timing['last_service'],
                    'avg_frequency_minutes': timing['avg_frequency_minutes']
                }
        
        # Convert to list
        station_data = list(station_dict.values())
        
        # Sort by total services
        station_data.sort(key=lambda x: x['total_services'], reverse=True)
        return station_data[:n]
    
    def generate_report(self):
        """Generate comprehensive JSON report"""
        print("\nGenerating comprehensive report...")
        
        top_hubs = self.get_top_critical_hubs(10)
        busiest = self.get_busiest_stations(5)
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'Hyderabad Metro - Enhanced Analysis',
            'network_overview': {
                'stations': self.G.number_of_nodes(),
                'connections': self.G.number_of_edges(),
                'density': round(nx.density(self.G), 4)
            },
            'top_critical_hubs': top_hubs,
            'timing_summary': {
                'stations_with_timing_data': len(self.station_timings),
                'busiest_stations': busiest
            }
        }
        
        # Save report
        output_dir = Path("../analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "enhanced_metro_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {output_dir / 'enhanced_metro_report.json'}")
        return report
    
    def display_summary(self, report):
        """Display summary of key findings"""
        print("\n" + "="*70)
        print("METRO CRITICAL STATIONS - IMPORTANCE & TIMING ANALYSIS")
        print("="*70)
        
        print("\nTop 10 Critical Hubs:")
        print("-" * 70)
        for i, hub in enumerate(report['top_critical_hubs'], 1):
            print(f"\n{i}. {hub['name']}")
            print(f"   Betweenness: {hub['betweenness_centrality']:.4f} | Services: {hub['total_services']}")
            print(f"   Hours: {hub['first_service']} - {hub['last_service']} | Freq: {hub['avg_frequency_minutes']} min")
            print(f"   {hub['peak_hours']}")
        
        print("\n" + "="*70)
        print("TOP 5 BUSIEST STATIONS BY SERVICE FREQUENCY")
        print("="*70)
        for i, station in enumerate(report['timing_summary']['busiest_stations'], 1):
            print(f"\n{i}. {station['name']}")
            print(f"   {station['total_services']} services/day")
            print(f"   {station['first_service']} - {station['last_service']}")
            print(f"   Every {station['avg_frequency_minutes']} min")
    
    def run_analysis(self):
        """Run complete analysis"""
        self.build_network()
        self.compute_centrality()
        self.analyze_station_timings()
        report = self.generate_report()
        self.display_summary(report)
        return report

if __name__ == "__main__":
    print("="*70)
    print("ENHANCED HYDERABAD METRO NETWORK ANALYSIS")
    print("="*70)
    
    analyzer = EnhancedMetroAnalyzer()
    report = analyzer.run_analysis()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
