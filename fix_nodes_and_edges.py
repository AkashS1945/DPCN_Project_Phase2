#!/usr/bin/env python3
"""
Fix node naming conventions and ensure bidirectional edges.
1. Rename bus feeder nodes with directional suffixes (North, South, East, West)
2. Ensure all edges have reverse edges
"""

import pandas as pd
from collections import defaultdict

# Direction mapping based on position index (0, 1, 2, 3)
# Assuming: 0=North, 1=East, 2=South, 3=West (clockwise from top)
DIRECTIONS = {
    0: 'North',
    1: 'East', 
    2: 'South',
    3: 'West'
}

def fix_node_names():
    """Fix bus feeder node names to include directions."""
    print("Reading nodes.csv...")
    nodes = pd.read_csv('nodes.csv')
    
    # Group bus feeder nodes by their base station
    feeder_groups = defaultdict(list)
    
    for idx, row in nodes.iterrows():
        node_id = row['node_id']
        if 'Bus_feeder_' in node_id and '_Metro_' in node_id:
            # Extract base station (e.g., "AME_Metro" from "Bus_feeder_AME_Metro_0")
            parts = node_id.split('_')
            if len(parts) >= 4:
                base = '_'.join(parts[2:-1])  # e.g., "AME_Metro"
                position = int(parts[-1])  # e.g., 0, 1, 2, 3
                feeder_groups[base].append((idx, node_id, position))
    
    # Update names with directions
    updates = []
    for base, feeders in feeder_groups.items():
        station_name = base.replace('_Metro', '')
        for idx, node_id, position in feeders:
            direction = DIRECTIONS.get(position, str(position))
            new_name = f"Bus {direction} of {station_name}"
            old_name = nodes.at[idx, 'name']
            nodes.at[idx, 'name'] = new_name
            updates.append(f"  {node_id}: '{old_name}' → '{new_name}'")
    
    print(f"\nUpdated {len(updates)} bus feeder node names:")
    for update in sorted(updates)[:20]:  # Show first 20
        print(update)
    if len(updates) > 20:
        print(f"  ... and {len(updates) - 20} more")
    
    # Save updated nodes
    nodes.to_csv('nodes.csv', index=False)
    print(f"\n✓ Saved updated nodes.csv with {len(nodes)} nodes")
    
    return nodes

def ensure_bidirectional_edges():
    """Ensure all edges have reverse edges."""
    print("\nReading edges.csv...")
    edges = pd.read_csv('edges.csv')
    
    print(f"Original edges: {len(edges)}")
    
    # Create edge lookup (from_id, to_id) -> edge data
    edge_lookup = {}
    for idx, row in edges.iterrows():
        key = (row['from_id'], row['to_id'])
        edge_lookup[key] = row.to_dict()
    
    # Find missing reverse edges
    missing_reverse = []
    for (from_id, to_id), edge_data in edge_lookup.items():
        reverse_key = (to_id, from_id)
        if reverse_key not in edge_lookup:
            missing_reverse.append({
                'from_id': to_id,
                'to_id': from_id,
                'mode': edge_data['mode'],
                'intra_or_inter': edge_data['intra_or_inter'],
                'reason': edge_data['reason'],
                'distance_km': edge_data['distance_km'],
                'time_base_min': edge_data['time_base_min'],
                'time_min': edge_data['time_min'],
                'cost_base_rs': edge_data['cost_base_rs'],
                'cost_rs': edge_data['cost_rs'],
                'region_from': edge_data['region_to'],
                'region_to': edge_data['region_from']
            })
    
    if missing_reverse:
        print(f"\nFound {len(missing_reverse)} missing reverse edges:")
        for i, edge in enumerate(missing_reverse[:10]):  # Show first 10
            print(f"  {edge['from_id']} → {edge['to_id']} ({edge['mode']})")
        if len(missing_reverse) > 10:
            print(f"  ... and {len(missing_reverse) - 10} more")
        
        # Add missing reverse edges
        new_edges = pd.DataFrame(missing_reverse)
        edges = pd.concat([edges, new_edges], ignore_index=True)
        
        # Save updated edges
        edges.to_csv('edges.csv', index=False)
        print(f"\n✓ Added {len(missing_reverse)} reverse edges")
        print(f"✓ Total edges now: {len(edges)}")
    else:
        print("\n✓ All edges already have reverse edges!")
    
    return edges

def verify_changes():
    """Verify the changes made."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    nodes = pd.read_csv('nodes.csv')
    edges = pd.read_csv('edges.csv')
    
    # Check bus feeder nodes
    bus_feeders = nodes[nodes['node_id'].str.contains('Bus_feeder_', na=False)]
    print(f"\nBus feeder nodes: {len(bus_feeders)}")
    print("\nSample of renamed nodes:")
    for _, row in bus_feeders.head(12).iterrows():
        print(f"  {row['node_id']}: {row['name']}")
    
    # Check edge symmetry
    edge_pairs = defaultdict(list)
    for _, row in edges.iterrows():
        key = tuple(sorted([row['from_id'], row['to_id']]))
        edge_pairs[key].append((row['from_id'], row['to_id']))
    
    asymmetric = [k for k, v in edge_pairs.items() if len(v) != 2]
    if asymmetric:
        print(f"\n⚠ Warning: {len(asymmetric)} asymmetric edge pairs found:")
        for pair in asymmetric[:5]:
            directions = edge_pairs[pair]
            print(f"  {pair}: {directions}")
    else:
        print(f"\n✓ All {len(edge_pairs)} edge pairs are bidirectional!")
    
    print(f"\nFinal counts:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Edge pairs: {len(edge_pairs)}")

def main():
    print("="*60)
    print("FIXING NODE NAMES AND EDGES")
    print("="*60)
    
    # Fix node names
    nodes = fix_node_names()
    
    # Ensure bidirectional edges
    edges = ensure_bidirectional_edges()
    
    # Verify changes
    verify_changes()
    
    print("\n" + "="*60)
    print("✓ DONE! All fixes applied successfully.")
    print("="*60)

if __name__ == '__main__':
    main()
