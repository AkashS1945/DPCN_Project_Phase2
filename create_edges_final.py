#!/usr/bin/env python3
"""
Create edges with proper line connections for Metro and MMTS.
Ensures all line segments are connected properly.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from itertools import combinations

# Metro Line Structure (based on actual Hyderabad Metro GTFS data)
METRO_LINES = {
    'Red': [
        # Miyapur to LB Nagar (Main Red Line - Corridor I)
        'MYP', 'JNT', 'KPH', 'KUK', 'BLR', 'MSP', 'BTN', 'ERA', 'ESI', 'SRN',
        'AME', 'PUN', 'IRM', 'KHA', 'LKP', 'ASM', 'NAM', 'GAB', 'OMC', 'MGB',
        'MKL', 'NEM', 'MSB', 'DSN', 'CHP', 'VOM', 'LBN'
    ],
    'Blue': [
        # Raidurg to Nagole (Blue Line - Corridor II) - Extended to include all stations
        'RDG', 'HTC', 'DGC', 'MAD', 'PED', 'JCP', 'JR5', 'YUG', 'MUN', 'AME',
        'BEG', 'PRN', 'ROP', 'PAR', 'PRG', 'SEC_E', 'MET', 'TAR', 'HSG', 'NGR',
        'STD', 'UPL', 'NAG'
    ],
    'Green': [
        # JBS to MGBS (Green Line - Corridor III) - Correct sequence from GTFS
        'JBS', 'SCR', 'GNH', 'MSH', 'RTC', 'CDP', 'NAR', 'SUB', 'MGB'
    ]
}

# MMTS corridors
MMTS_CORRIDORS = {
    'Main': [
        'LPI', 'CDNR', 'HFZ', 'HTCY', 'BRBD', 'BTNR', 'FNB', 'NCHS', 'NLRD', 'KQD',
        'LKPL', 'BMT', 'SC', 'ATC', 'JOO', 'VAR', 'STPD', 'MJF', 'CHZ'
    ],
    'South': [
        'SC', 'JET', 'SJVP', 'HYB', 'KCG', 'MXT', 'HPG', 'YKA', 'DQR', 'FM', 'NSVP', 'BDVL'
    ],
    'North': [
        'SC', 'LGDH', 'DYE', 'SFX', 'RKO', 'AMQ', 'CVB', 'ALW', 'BOZ', 'BMO', 'GWV', 'MED'
    ],
    'RCPT_Branch': ['RCPT', 'BHEL', 'TLPR', 'LPI'],  # Connects to Main at LPI
    'GDPL_Branch': ['GDPL', 'GWV']  # Connects to North at GWV
}


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km."""
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def compute_time(distance_km, mode, node_type='regular'):
    """Compute base travel time."""
    if mode == 'metro':
        return 2.0 + distance_km * 1.8
    elif mode == 'mmts':
        return 3.0 + distance_km * 2.0
    elif mode == 'bus':
        if node_type == 'express':
            return 1.5 + distance_km * 1.714  # Trunk/express
        return 2.0 + distance_km * 3.0  # Local
    elif mode == 'auto':
        return 3.0 + distance_km * 3.75
    elif mode == 'transfer' or mode == 'walk':
        return distance_km * 12.0
    return distance_km * 5.0


def compute_cost(distance_km, mode, region_from, region_to):
    """Compute base cost."""
    if mode == 'metro':
        base = 10 + distance_km * 2.0
    elif mode == 'mmts':
        base = 5 + distance_km * 1.5
    elif mode == 'bus':
        base = 5 + distance_km * 1.5
    elif mode == 'auto':
        base = 30 + distance_km * 13.0
    else:  # walk/transfer
        base = 0
    
    # Apply regional modifier
    Df_avg = (region_from['Df'] + region_to['Df']) / 2
    Hf_avg = (region_from['Hf'] + region_to['Hf']) / 2
    modifier = 1 + 0.1 * (Df_avg + Hf_avg - 2)
    
    return base * modifier


def apply_time_modifier(time_base, region_from, region_to):
    """Apply congestion modifier to time."""
    Cf_avg = (region_from['Cf'] + region_to['Cf']) / 2
    return time_base * Cf_avg


def create_metro_edges(nodes_df):
    """Create Metro edges following line structure."""
    print("\n🚇 Creating Metro edges...")
    edges = []
    metro_nodes = nodes_df[nodes_df['layer'] == 'metro'].copy()
    node_lookup = {row['node_id']: row for _, row in metro_nodes.iterrows()}
    regions_df = pd.read_csv('regions.csv')
    
    # Build edges for each line
    for line_name, stations in METRO_LINES.items():
        print(f"   Processing {line_name} Line...")
        connected = 0
        for i in range(len(stations) - 1):
            from_id = f"{stations[i]}_Metro"
            to_id = f"{stations[i+1]}_Metro"
            
            if from_id not in node_lookup or to_id not in node_lookup:
                print(f"      ⚠️  Skipping {stations[i]} → {stations[i+1]} (nodes not found)")
                continue
            
            from_node = node_lookup[from_id]
            to_node = node_lookup[to_id]
            
            dist = haversine(from_node['lat'], from_node['lon'], to_node['lat'], to_node['lon'])
            dist = max(dist, 0.5)  # Min 0.5 km between stations
            
            time_base = compute_time(dist, 'metro')
            
            region_from = regions_df[regions_df['id'] == from_node['region']].iloc[0]
            region_to = regions_df[regions_df['id'] == to_node['region']].iloc[0]
            
            time_final = apply_time_modifier(time_base, region_from, region_to)
            cost_final = compute_cost(dist, 'metro', region_from, region_to)
            
            # Bidirectional
            for direction in [(from_id, to_id), (to_id, from_id)]:
                edges.append({
                    'from_id': direction[0],
                    'to_id': direction[1],
                    'mode': 'metro',
                    'intra_or_inter': 'intra',
                    'reason': f'{line_name}_Line',
                    'distance_km': dist,
                    'time_base_min': time_base,
                    'time_min': time_final,
                    'cost_base_rs': cost_final,
                    'cost_rs': cost_final,
                    'region_from': from_node['region'],
                    'region_to': to_node['region']
                })
            connected += 2
        print(f"      ✓ {connected} edges for {line_name} Line")
    
    # Ensure ALL metro nodes have at least 2 connections
    metro_edge_count = {}
    for edge in edges:
        for node_id in [edge['from_id'], edge['to_id']]:
            metro_edge_count[node_id] = metro_edge_count.get(node_id, 0) + 1
    
    # Find isolated metro nodes and connect to nearest
    for node_id, row in metro_nodes.iterrows():
        if metro_edge_count.get(row['node_id'], 0) < 2:
            print(f"   ⚠️  {row['name']} has only {metro_edge_count.get(row['node_id'], 0)} edges, connecting to nearest...")
            # Find nearest metro node with edges
            min_dist = float('inf')
            nearest = None
            for other_id, other_row in metro_nodes.iterrows():
                if other_row['node_id'] == row['node_id']:
                    continue
                dist = haversine(row['lat'], row['lon'], other_row['lat'], other_row['lon'])
                if dist < min_dist and metro_edge_count.get(other_row['node_id'], 0) > 0:
                    min_dist = dist
                    nearest = other_row
            
            if nearest is not None:
                region_from = regions_df[regions_df['id'] == row['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == nearest['region']].iloc[0]
                time_base = compute_time(min_dist, 'metro')
                time_final = apply_time_modifier(time_base, region_from, region_to)
                cost_final = compute_cost(min_dist, 'metro', region_from, region_to)
                
                for direction in [(row['node_id'], nearest['node_id']), (nearest['node_id'], row['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'metro',
                        'intra_or_inter': 'intra',
                        'reason': 'connectivity_fix',
                        'distance_km': min_dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': cost_final,
                        'cost_rs': cost_final,
                        'region_from': row['region'],
                        'region_to': nearest['region']
                    })
    
    print(f"   ✓ Created {len(edges)} Metro edges total")
    return edges


def create_mmts_edges(nodes_df):
    """Create MMTS edges following corridor structure."""
    print("\n🚂 Creating MMTS edges...")
    edges = []
    mmts_nodes = nodes_df[nodes_df['layer'] == 'mmts'].copy()
    node_lookup = {row['node_id']: row for _, row in mmts_nodes.iterrows()}
    regions_df = pd.read_csv('regions.csv')
    
    for corridor_name, stations in MMTS_CORRIDORS.items():
        print(f"   Processing {corridor_name} corridor...")
        connected = 0
        for i in range(len(stations) - 1):
            from_id = f"{stations[i]}_MMTS"
            to_id = f"{stations[i+1]}_MMTS"
            
            if from_id not in node_lookup or to_id not in node_lookup:
                print(f"      ⚠️  Skipping {stations[i]} → {stations[i+1]} (nodes not found)")
                continue
            
            from_node = node_lookup[from_id]
            to_node = node_lookup[to_id]
            
            dist = haversine(from_node['lat'], from_node['lon'], to_node['lat'], to_node['lon'])
            dist = max(dist, 0.5)
            
            time_base = compute_time(dist, 'mmts')
            
            region_from = regions_df[regions_df['id'] == from_node['region']].iloc[0]
            region_to = regions_df[regions_df['id'] == to_node['region']].iloc[0]
            
            time_final = apply_time_modifier(time_base, region_from, region_to)
            cost_final = compute_cost(dist, 'mmts', region_from, region_to)
            
            # Bidirectional
            for direction in [(from_id, to_id), (to_id, from_id)]:
                edges.append({
                    'from_id': direction[0],
                    'to_id': direction[1],
                    'mode': 'mmts',
                    'intra_or_inter': 'intra',
                    'reason': f'{corridor_name}_corridor',
                    'distance_km': dist,
                    'time_base_min': time_base,
                    'time_min': time_final,
                    'cost_base_rs': cost_final,
                    'cost_rs': cost_final,
                    'region_from': from_node['region'],
                    'region_to': to_node['region']
                })
            connected += 2
        print(f"      ✓ {connected} edges for {corridor_name}")
    
    # Ensure ALL MMTS nodes have at least 2 connections
    mmts_edge_count = {}
    for edge in edges:
        for node_id in [edge['from_id'], edge['to_id']]:
            mmts_edge_count[node_id] = mmts_edge_count.get(node_id, 0) + 1
    
    for node_id, row in mmts_nodes.iterrows():
        if mmts_edge_count.get(row['node_id'], 0) < 2:
            print(f"   ⚠️  {row['name']} has only {mmts_edge_count.get(row['node_id'], 0)} edges, connecting to nearest...")
            # Find 2 nearest MMTS nodes
            distances = []
            for other_id, other_row in mmts_nodes.iterrows():
                if other_row['node_id'] == row['node_id']:
                    continue
                dist = haversine(row['lat'], row['lon'], other_row['lat'], other_row['lon'])
                distances.append((dist, other_row))
            
            distances.sort(key=lambda x: x[0])
            for dist, nearest in distances[:2]:
                region_from = regions_df[regions_df['id'] == row['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == nearest['region']].iloc[0]
                time_base = compute_time(dist, 'mmts')
                time_final = apply_time_modifier(time_base, region_from, region_to)
                cost_final = compute_cost(dist, 'mmts', region_from, region_to)
                
                for direction in [(row['node_id'], nearest['node_id']), (nearest['node_id'], row['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'mmts',
                        'intra_or_inter': 'intra',
                        'reason': 'connectivity_fix',
                        'distance_km': dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': cost_final,
                        'cost_rs': cost_final,
                        'region_from': row['region'],
                        'region_to': nearest['region']
                    })
    
    print(f"   ✓ Created {len(edges)} MMTS edges total")
    return edges


def create_bus_edges(nodes_df):
    """Create bus edges connecting nearby bus stops - MINIMUM 4 edges per bus node."""
    print("\n🚌 Creating Bus edges...")
    bus_nodes = nodes_df[nodes_df['layer'] == 'bus'].copy()
    edges = []
    regions_df = pd.read_csv('regions.csv')
    
    print(f"   Processing {len(bus_nodes)} bus nodes...")
    for idx, from_bus in bus_nodes.iterrows():
        # Find ALL buses and sort by distance
        distances = []
        for _, to_bus in bus_nodes.iterrows():
            if from_bus['node_id'] == to_bus['node_id']:
                continue
            dist = haversine(from_bus['lat'], from_bus['lon'], to_bus['lat'], to_bus['lon'])
            distances.append((dist, to_bus))
        
        # Sort by distance and ensure minimum 4 connections
        distances.sort(key=lambda x: x[0])
        # Guarantee minimum 4 connections (or all available if less than 4)
        min_connections = min(4, len(distances))
        # But prefer nearby ones within 5km if available
        nearby = [d for d in distances if d[0] < 5.0]
        if len(nearby) >= 4:
            min_connections = max(min_connections, min(6, len(nearby)))  # Connect to up to 6 if nearby
        
        for dist, to_bus in distances[:min_connections]:
            time_base = compute_time(dist, 'bus', from_bus['type'])
            
            region_from = regions_df[regions_df['id'] == from_bus['region']].iloc[0]
            region_to = regions_df[regions_df['id'] == to_bus['region']].iloc[0]
            
            time_final = apply_time_modifier(time_base, region_from, region_to)
            cost_final = compute_cost(dist, 'bus', region_from, region_to)
            
            # Unidirectional to avoid duplicates (other node will add reverse)
            edges.append({
                'from_id': from_bus['node_id'],
                'to_id': to_bus['node_id'],
                'mode': 'bus',
                'intra_or_inter': 'intra',
                'reason': 'bus_route',
                'distance_km': dist,
                'time_base_min': time_base,
                'time_min': time_final,
                'cost_base_rs': cost_final,
                'cost_rs': cost_final,
                'region_from': from_bus['region'],
                'region_to': to_bus['region']
            })
    
    print(f"   ✓ Created {len(edges)} Bus edges (avg {len(edges)/len(bus_nodes):.1f} edges per bus)")
    return edges


def create_auto_edges(nodes_df):
    """Create auto edges - MINIMUM 3 edges per auto stand."""
    print("\n🚕 Creating Auto edges...")
    auto_nodes = nodes_df[nodes_df['layer'] == 'auto'].copy()
    edges = []
    regions_df = pd.read_csv('regions.csv')
    
    print(f"   Processing {len(auto_nodes)} auto stands...")
    for idx, from_auto in auto_nodes.iterrows():
        # Find ALL autos and sort by distance
        distances = []
        for _, to_auto in auto_nodes.iterrows():
            if from_auto['node_id'] == to_auto['node_id']:
                continue
            dist = haversine(from_auto['lat'], from_auto['lon'], to_auto['lat'], to_auto['lon'])
            distances.append((dist, to_auto))
        
        # Sort by distance and connect to at least 3 nearest
        distances.sort(key=lambda x: x[0])
        # Ensure minimum 3 connections (or all available if less than 3)
        min_connections = min(3, len(distances))
        # But prefer nearby ones within 3km if available
        nearby = [d for d in distances if d[0] < 3.0]
        if len(nearby) >= 3:
            min_connections = max(min_connections, min(5, len(nearby)))  # Connect to up to 5 if nearby
        
        for dist, to_auto in distances[:min_connections]:
            time_base = compute_time(dist, 'auto')
            
            region_from = regions_df[regions_df['id'] == from_auto['region']].iloc[0]
            region_to = regions_df[regions_df['id'] == to_auto['region']].iloc[0]
            
            time_final = apply_time_modifier(time_base, region_from, region_to)
            cost_final = compute_cost(dist, 'auto', region_from, region_to)
            
            # Unidirectional to avoid duplicates
            edges.append({
                'from_id': from_auto['node_id'],
                'to_id': to_auto['node_id'],
                'mode': 'auto',
                'intra_or_inter': 'intra',
                'reason': 'auto_network',
                'distance_km': dist,
                'time_base_min': time_base,
                'time_min': time_final,
                'cost_base_rs': cost_final,
                'cost_rs': cost_final,
                'region_from': from_auto['region'],
                'region_to': to_auto['region']
            })
    
    print(f"   ✓ Created {len(edges)} Auto edges (avg {len(edges)/len(auto_nodes):.1f} edges per auto)")
    return edges


def create_transfer_edges(nodes_df):
    """Create all inter-layer transfer edges."""
    print("\n🔄 Creating Transfer edges...")
    edges = []
    regions_df = pd.read_csv('regions.csv')
    
    metro_nodes = nodes_df[nodes_df['layer'] == 'metro']
    mmts_nodes = nodes_df[nodes_df['layer'] == 'mmts']
    bus_nodes = nodes_df[nodes_df['layer'] == 'bus']
    auto_nodes = nodes_df[nodes_df['layer'] == 'auto']
    
    # Metro → Bus (within 1 km)
    count = 0
    for _, metro in metro_nodes.iterrows():
        for _, bus in bus_nodes.iterrows():
            dist = haversine(metro['lat'], metro['lon'], bus['lat'], bus['lon'])
            if dist < 1.0:
                time_base = compute_time(dist, 'walk')
                region_from = regions_df[regions_df['id'] == metro['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == bus['region']].iloc[0]
                time_final = apply_time_modifier(time_base, region_from, region_to)
                
                for direction in [(metro['node_id'], bus['node_id']), (bus['node_id'], metro['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'transfer',
                        'intra_or_inter': 'inter',
                        'reason': 'metro_bus_transfer',
                        'distance_km': dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': 0,
                        'cost_rs': 0,
                        'region_from': metro['region'],
                        'region_to': bus['region']
                    })
                count += 2
    print(f"   ✓ Metro↔Bus: {count} transfers")
    
    # Metro → Auto (within 0.8 km)
    count = 0
    for _, metro in metro_nodes.iterrows():
        for _, auto in auto_nodes.iterrows():
            dist = haversine(metro['lat'], metro['lon'], auto['lat'], auto['lon'])
            if dist < 0.8:
                time_base = compute_time(dist, 'walk')
                region_from = regions_df[regions_df['id'] == metro['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == auto['region']].iloc[0]
                time_final = apply_time_modifier(time_base, region_from, region_to)
                
                for direction in [(metro['node_id'], auto['node_id']), (auto['node_id'], metro['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'transfer',
                        'intra_or_inter': 'inter',
                        'reason': 'metro_auto_transfer',
                        'distance_km': dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': 0,
                        'cost_rs': 0,
                        'region_from': metro['region'],
                        'region_to': auto['region']
                    })
                count += 2
    print(f"   ✓ Metro↔Auto: {count} transfers")
    
    # MMTS → Bus
    count = 0
    for _, mmts in mmts_nodes.iterrows():
        for _, bus in bus_nodes.iterrows():
            dist = haversine(mmts['lat'], mmts['lon'], bus['lat'], bus['lon'])
            if dist < 1.0:
                time_base = compute_time(dist, 'walk')
                region_from = regions_df[regions_df['id'] == mmts['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == bus['region']].iloc[0]
                time_final = apply_time_modifier(time_base, region_from, region_to)
                
                for direction in [(mmts['node_id'], bus['node_id']), (bus['node_id'], mmts['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'transfer',
                        'intra_or_inter': 'inter',
                        'reason': 'mmts_bus_transfer',
                        'distance_km': dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': 0,
                        'cost_rs': 0,
                        'region_from': mmts['region'],
                        'region_to': bus['region']
                    })
                count += 2
    print(f"   ✓ MMTS↔Bus: {count} transfers")
    
    # MMTS → Auto
    count = 0
    for _, mmts in mmts_nodes.iterrows():
        for _, auto in auto_nodes.iterrows():
            dist = haversine(mmts['lat'], mmts['lon'], auto['lat'], auto['lon'])
            if dist < 0.8:
                time_base = compute_time(dist, 'walk')
                region_from = regions_df[regions_df['id'] == mmts['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == auto['region']].iloc[0]
                time_final = apply_time_modifier(time_base, region_from, region_to)
                
                for direction in [(mmts['node_id'], auto['node_id']), (auto['node_id'], mmts['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'transfer',
                        'intra_or_inter': 'inter',
                        'reason': 'mmts_auto_transfer',
                        'distance_km': dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': 0,
                        'cost_rs': 0,
                        'region_from': mmts['region'],
                        'region_to': auto['region']
                    })
                count += 2
    print(f"   ✓ MMTS↔Auto: {count} transfers")
    
    # Bus → Auto
    count = 0
    for _, bus in bus_nodes.iterrows():
        for _, auto in auto_nodes.iterrows():
            dist = haversine(bus['lat'], bus['lon'], auto['lat'], auto['lon'])
            if dist < 0.3:
                time_base = compute_time(dist, 'walk')
                region_from = regions_df[regions_df['id'] == bus['region']].iloc[0]
                region_to = regions_df[regions_df['id'] == auto['region']].iloc[0]
                time_final = apply_time_modifier(time_base, region_from, region_to)
                
                for direction in [(bus['node_id'], auto['node_id']), (auto['node_id'], bus['node_id'])]:
                    edges.append({
                        'from_id': direction[0],
                        'to_id': direction[1],
                        'mode': 'transfer',
                        'intra_or_inter': 'inter',
                        'reason': 'bus_auto_transfer',
                        'distance_km': dist,
                        'time_base_min': time_base,
                        'time_min': time_final,
                        'cost_base_rs': 0,
                        'cost_rs': 0,
                        'region_from': bus['region'],
                        'region_to': auto['region']
                    })
                count += 2
    print(f"   ✓ Bus↔Auto: {count} transfers")
    
    print(f"   ✓ Total transfer edges: {len(edges)}")
    return edges


def main():
    """Main execution."""
    print("=" * 70)
    print("🔗 CREATING EDGES WITH PROPER LINE CONNECTIONS")
    print("=" * 70)
    
    # Load nodes
    nodes_df = pd.read_csv('nodes.csv')
    print(f"\n📊 Loaded {len(nodes_df)} nodes")
    
    all_edges = []
    
    # Create edges
    all_edges.extend(create_metro_edges(nodes_df))
    all_edges.extend(create_mmts_edges(nodes_df))
    all_edges.extend(create_bus_edges(nodes_df))
    all_edges.extend(create_auto_edges(nodes_df))
    all_edges.extend(create_transfer_edges(nodes_df))
    
    # Save
    edges_df = pd.DataFrame(all_edges)
    edges_df.to_csv('edges.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ EDGE CREATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Edge Summary:")
    print(edges_df.groupby('mode').size())
    print(f"\nTotal edges: {len(edges_df)}")
    print(f"\n💾 Saved to edges.csv")


if __name__ == "__main__":
    main()
