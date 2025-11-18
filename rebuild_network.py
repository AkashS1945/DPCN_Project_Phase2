#!/usr/bin/env python3
"""
Complete rebuild of Hyderabad Multimodal Network with fixes:
1. Remove non-Hyderabad MMTS stations (Bibinagar, Jangaon, Bhongir, etc.)
2. Add all Metro line connections (Red, Blue, Green)
3. Fix MMTS connectivity
4. Regenerate all CSVs
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans

# Hyderabad city bounds (approximately)
HYD_LAT_MIN, HYD_LAT_MAX = 17.2, 17.7
HYD_LON_MIN, HYD_LON_MAX = 78.2, 78.7

# MMTS stations to EXCLUDE (outside Hyderabad)
EXCLUDE_MMTS = [
    'BN', 'BIBINAGAR',  # Bibinagar - 78.79°E (too far east)
    'ZN', 'JANGAON',    # Jangaon - 79.15°E (way outside)
    'BG', 'BHONGIR',    # Bhongir - 78.89°E (outside)
    'ALER',             # Aler - 79.04°E (outside)
    'RAG', 'RAIGIR',    # Raigir - 78.94°E (outside)
    'PBP', 'PEMBARTI',  # Pembarti - 79.11°E (outside)
    'WP', 'WANGAPALLI', # Wangapalli - 78.98°E (outside)
    'GT', 'GHATKESAR'   # Ghatkesar - 78.67°E (borderline, but exclude)
]

# Metro Line Structure - CORRECTED per official map
METRO_LINES = {
    'Red': [
        # Miyapur to LB Nagar (Red Line - Corridor I)
        'MYP', 'JNT', 'KPH', 'KUK', 'BLR', 'VOM', 'MSB', 'MSP', 'DSN', 'NEM',
        'OMC', 'KHA', 'ERA', 'SRN', 'ESI', 'CHP', 'ASM', 'GAB', 'PUN', 'IRM',
        'AME', 'MGB', 'NAM', 'LKP', 'BTN', 'MKL', 'LBN',
        # Red line also goes through JBS (interchange with Green)
        'JBS'
    ],
    'Blue': [
        # Raidurg to Nagole (Blue Line - Corridor II) - Complete sequence
        'RDG', 'HTC', 'DGC', 'MAD', 'PED', 'JCP', 'JR5', 'YUG', 'MUN', 'AME',
        'BEG', 'PRN', 'ROP', 'PAR', 'PRG', 'SEC_E', 'MET', 'TAR', 'HSG', 'NGR',
        'STD', 'UPL', 'NAG'
    ],
    'Green': [
        # JBS to MGBS (Green Line - Corridor III) - Correct sequence from official map
        'JBS', 'SCR', 'GNH', 'MSH', 'RTC', 'CDP', 'NAR', 'SUB', 'MGB'
    ]
}

# MMTS Line Structure (main corridor + branches)
MMTS_MAIN_LINE = [
    'LPI', 'CDNR', 'HFZ', 'HTCY', 'BRBD', 'BTNR', 'FNB', 'NCHS', 'NLRD', 'KQD',
    'LKPL', 'BMT', 'SC', 'ATC', 'JOO', 'VAR', 'STPD', 'MJF', 'CHZ'
]

MMTS_SOUTH_BRANCH = [
    'SC', 'JET', 'SJVP', 'HYB', 'KCG', 'MXT', 'HPG', 'YKA', 'DQR', 'FM', 'NSVP', 'BDVL'
]

MMTS_NORTH_BRANCH = [
    'SC', 'LGDH', 'DYE', 'SFX', 'RKO', 'AMQ', 'CVB', 'ALW', 'BOZ', 'BMO', 'GWV', 'MED'
]

# Regions
REGIONS = [
    {'id': 'R1', 'name': 'Secunderabad/North-Central', 'center': (17.444, 78.501), 'Df': 1.4, 'Hf': 1.3, 'Cf': 1.3},
    {'id': 'R2', 'name': 'Old City/South', 'center': (17.375, 78.474), 'Df': 1.5, 'Hf': 1.2, 'Cf': 1.4},
    {'id': 'R3', 'name': 'Deccan/South-Central', 'center': (17.424, 78.474), 'Df': 1.2, 'Hf': 1.1, 'Cf': 1.2},
    {'id': 'R4', 'name': 'HITEC City/West', 'center': (17.448, 78.392), 'Df': 1.3, 'Hf': 1.2, 'Cf': 1.2},
    {'id': 'R5', 'name': 'Kukatpally/Northwest', 'center': (17.492, 78.403), 'Df': 1.1, 'Hf': 1.0, 'Cf': 1.1},
    {'id': 'R6', 'name': 'Uppal/East', 'center': (17.407, 78.560), 'Df': 0.9, 'Hf': 1.0, 'Cf': 1.0},
    {'id': 'R7', 'name': 'LB Nagar/Periphery', 'center': (17.350, 78.550), 'Df': 0.8, 'Hf': 1.0, 'Cf': 1.0}
]

# Primary bus hubs
PRIMARY_HUBS = [
    ('Secunderabad', 17.444, 78.501), ('MGBS', 17.380, 78.486), ('Ameerpet', 17.436, 78.445),
    ('Dilsukhnagar', 17.369, 78.525), ('Kukatpally', 17.485, 78.412), ('LB Nagar', 17.350, 78.553),
    ('Uppal', 17.407, 78.560), ('Lakdi-ka-pul', 17.404, 78.465), ('Jubilee Hills', 17.426, 78.409),
    ('Mehdipatnam', 17.393, 78.435), ('Charminar', 17.361, 78.475), ('Afzalgunj', 17.383, 78.479),
    ('Koti', 17.385, 78.480), ('Nampally', 17.392, 78.470), ('Panjagutta', 17.429, 78.451),
    ('SR Nagar', 17.443, 78.441), ('Erragadda', 17.457, 78.434), ('Begumpet', 17.439, 78.468),
    ('Paradise', 17.438, 78.499), ('ECIL', 17.479, 78.571), ('Habsiguda', 17.422, 78.537),
    ('Tarnaka', 17.423, 78.524), ('Musheerabad', 17.427, 78.485), ('Kacheguda', 17.390, 78.500),
    ('Malakpet', 17.377, 78.494), ('Moosapet', 17.472, 78.426), ('KPHB', 17.494, 78.402),
    ('Miyapur', 17.497, 78.373), ('HITEC City', 17.448, 78.381), ('Gachibowli', 17.440, 78.363)
]


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def assign_region(lat, lon):
    """Assign node to nearest region."""
    min_dist = float('inf')
    best_region = REGIONS[0]
    for region in REGIONS:
        dist = haversine(lat, lon, region['center'][0], region['center'][1])
        if dist < min_dist:
            min_dist = dist
            best_region = region
    return best_region


def load_metro_stations():
    """Load Metro stations from GTFS."""
    df = pd.read_csv('Metro_Hyderabd/stops.txt')
    # Get parent stations only (location_type = 1)
    stations = df[df['location_type'] == 1].copy()
    
    # Determine metro line for each station
    metro_nodes = []
    for _, row in stations.iterrows():
        station_id = row['stop_id']
        
        # Determine which line(s) this station is on
        lines = []
        if station_id in METRO_LINES['Red']:
            lines.append('Red')
        if station_id in METRO_LINES['Blue']:
            lines.append('Blue')
        if station_id in METRO_LINES['Green']:
            lines.append('Green')
        
        line_str = '+'.join(lines) if lines else 'Red'  # Default to Red if not found
        
        region = assign_region(row['stop_lat'], row['stop_lon'])
        
        metro_nodes.append({
            'node_id': f"{station_id}_Metro",
            'name': row['stop_name'],
            'lat': row['stop_lat'],
            'lon': row['stop_lon'],
            'layer': 'metro',
            'type': 'station',
            'region': region['id'],
            'Df': region['Df'],
            'Hf': region['Hf'],
            'Cf': region['Cf'],
            'metro_line': line_str
        })
    
    return pd.DataFrame(metro_nodes)


def load_mmts_stations():
    """Load MMTS stations from GTFS, excluding non-Hyderabad stations."""
    df = pd.read_csv('Open_Data_MMTS_Hyd/stops.txt')
    
    mmts_nodes = []
    for _, row in df.iterrows():
        station_id = row['stop_id']
        
        # SKIP excluded stations
        if station_id in EXCLUDE_MMTS:
            print(f"   Excluding MMTS: {row['stop_name']} ({station_id}) - outside Hyderabad")
            continue
        
        # SKIP if outside city bounds
        if not (HYD_LAT_MIN <= row['stop_lat'] <= HYD_LAT_MAX and 
                HYD_LON_MIN <= row['stop_lon'] <= HYD_LON_MAX):
            print(f"   Excluding MMTS: {row['stop_name']} - outside bounds")
            continue
        
        region = assign_region(row['stop_lat'], row['stop_lon'])
        
        mmts_nodes.append({
            'node_id': f"{station_id}_MMTS",
            'name': row['stop_name'],
            'lat': row['stop_lat'],
            'lon': row['stop_lon'],
            'layer': 'mmts',
            'type': 'station',
            'region': region['id'],
            'Df': region['Df'],
            'Hf': region['Hf'],
            'Cf': region['Cf'],
            'metro_line': ''
        })
    
    return pd.DataFrame(mmts_nodes)


def create_bus_nodes(metro_df, mmts_df, target_count=200):
    """Create bus stops using hybrid method."""
    print(f"\n📍 Creating {target_count} bus stops...")
    
    bus_nodes = []
    
    # 1. Primary hubs
    for name, lat, lon in PRIMARY_HUBS:
        region = assign_region(lat, lon)
        bus_nodes.append({
            'node_id': f"Bus_{name.replace(' ', '_')}",
            'name': f"{name} Bus Hub",
            'lat': lat,
            'lon': lon,
            'layer': 'bus',
            'type': 'hub',
            'region': region['id'],
            'Df': region['Df'],
            'Hf': region['Hf'],
            'Cf': region['Cf'],
            'metro_line': ''
        })
    
    # 2. Feeder stops near Metro/MMTS
    all_rail = pd.concat([metro_df, mmts_df])
    for _, station in all_rail.iterrows():
        for i, angle in enumerate([0, 90, 180, 270]):
            offset_lat = station['lat'] + 0.003 * np.cos(np.radians(angle))
            offset_lon = station['lon'] + 0.003 * np.sin(np.radians(angle))
            
            region = assign_region(offset_lat, offset_lon)
            bus_nodes.append({
                'node_id': f"Bus_feeder_{station['node_id']}_{i}",
                'name': f"Bus near {station['name']}",
                'lat': offset_lat,
                'lon': offset_lon,
                'layer': 'bus',
                'type': 'feeder',
                'region': region['id'],
                'Df': region['Df'],
                'Hf': region['Hf'],
                'Cf': region['Cf'],
                'metro_line': ''
            })
    
    # 3. K-means clustering to reach target
    df_temp = pd.DataFrame(bus_nodes)
    if len(df_temp) < target_count:
        remaining = target_count - len(df_temp)
        X = df_temp[['lat', 'lon']].values
        kmeans = KMeans(n_clusters=remaining, random_state=42, n_init=10)
        kmeans.fit(X)
        
        for i, center in enumerate(kmeans.cluster_centers_):
            region = assign_region(center[0], center[1])
            bus_nodes.append({
                'node_id': f"Bus_cluster_{i}",
                'name': f"Bus Stop Cluster {i}",
                'lat': center[0],
                'lon': center[1],
                'layer': 'bus',
                'type': 'local',
                'region': region['id'],
                'Df': region['Df'],
                'Hf': region['Hf'],
                'Cf': region['Cf'],
                'metro_line': ''
            })
    
    # Select exactly target_count
    bus_df = pd.DataFrame(bus_nodes).head(target_count)
    print(f"   ✓ Created {len(bus_df)} bus nodes")
    return bus_df


def create_auto_nodes(metro_df, mmts_df, bus_df, target_count=400):
    """Create auto stands with regional distribution."""
    print(f"\n🚕 Creating ~{target_count} auto stands...")
    
    # Regional allocation
    allocation = {'R1': 0.18, 'R2': 0.14, 'R3': 0.17, 'R4': 0.20, 'R5': 0.12, 'R6': 0.10, 'R7': 0.09}
    
    auto_nodes = []
    all_locations = pd.concat([metro_df, mmts_df, bus_df])
    
    for region_id, percentage in allocation.items():
        region_target = int(target_count * percentage)
        region_data = REGIONS[[r['id'] for r in REGIONS].index(region_id)]
        
        # Priority locations
        region_locs = all_locations[all_locations['region'] == region_id]
        priority_locs = []
        for _, loc in region_locs.iterrows():
            for offset in range(2):
                offset_lat = loc['lat'] + np.random.uniform(-0.002, 0.002)
                offset_lon = loc['lon'] + np.random.uniform(-0.002, 0.002)
                priority_locs.append([offset_lat, offset_lon])
        
        # K-means clustering
        if priority_locs:
            X = np.array(priority_locs)
            k = min(region_target, len(priority_locs))
            if k > 0:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                
                for i, center in enumerate(kmeans.cluster_centers_):
                    auto_nodes.append({
                        'node_id': f"Auto_{region_id}_{i+1}",
                        'name': f"Auto Stand {region_id}-{i+1}",
                        'lat': center[0],
                        'lon': center[1],
                        'layer': 'auto',
                        'type': 'stand',
                        'region': region_id,
                        'Df': region_data['Df'],
                        'Hf': region_data['Hf'],
                        'Cf': region_data['Cf'],
                        'metro_line': ''
                    })
    
    auto_df = pd.DataFrame(auto_nodes)
    print(f"   ✓ Created {len(auto_df)} auto nodes")
    return auto_df


def main():
    """Main execution."""
    print("=" * 70)
    print("🔧 REBUILDING HYDERABAD MULTIMODAL NETWORK - FINAL VERSION")
    print("=" * 70)
    
    # Load Metro
    print("\n🚇 Loading Metro stations...")
    metro_df = load_metro_stations()
    print(f"   ✓ Loaded {len(metro_df)} Metro stations")
    
    # Load MMTS (filtered)
    print("\n🚂 Loading MMTS stations (filtering non-Hyderabad)...")
    mmts_df = load_mmts_stations()
    print(f"   ✓ Loaded {len(mmts_df)} MMTS stations (excluded {54 - len(mmts_df)} stations)")
    
    # Create bus
    bus_df = create_bus_nodes(metro_df, mmts_df, target_count=200)
    
    # Create auto
    auto_df = create_auto_nodes(metro_df, mmts_df, bus_df, target_count=400)
    
    # Combine
    all_nodes = pd.concat([metro_df, mmts_df, bus_df, auto_df], ignore_index=True)
    
    # Save
    print("\n💾 Saving nodes.csv...")
    all_nodes.to_csv('nodes.csv', index=False)
    print(f"   ✓ Saved {len(all_nodes)} nodes")
    
    # Save regions
    print("\n💾 Saving regions.csv...")
    regions_df = pd.DataFrame(REGIONS)
    regions_df['center_lat'] = regions_df['center'].apply(lambda x: x[0])
    regions_df['center_lon'] = regions_df['center'].apply(lambda x: x[1])
    regions_df = regions_df[['id', 'name', 'center_lat', 'center_lon', 'Df', 'Hf', 'Cf']]
    regions_df.to_csv('regions.csv', index=False)
    print(f"   ✓ Saved {len(regions_df)} regions")
    
    print("\n" + "=" * 70)
    print("✅ NETWORK REBUILD COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Node Counts:")
    print(f"   Metro:  {len(metro_df)}")
    print(f"   MMTS:   {len(mmts_df)}")
    print(f"   Bus:    {len(bus_df)}")
    print(f"   Auto:   {len(auto_df)}")
    print(f"   TOTAL:  {len(all_nodes)}")
    print("\n✨ Next: Run create_edges_final.py to generate edges with proper connections")


if __name__ == "__main__":
    main()
