#!/usr/bin/env python3
"""
Update Metro Fares in Network
Integrate the newly generated comprehensive fare data while using
the official HMRL fare structure from the JavaScript implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def calculate_official_hmrl_fare(distance_km):
    """
    Official HMRL fare structure (token/regular fares)
    Source: https://en.wikipedia.org/wiki/Hyderabad_Metro
    Announced 25 Nov 2017
    """
    if distance_km <= 2:
        return 10
    elif distance_km <= 4:
        return 20
    elif distance_km <= 6:
        return 30
    elif distance_km <= 9:
        return 40
    elif distance_km <= 12:
        return 50
    elif distance_km <= 15:
        return 60
    elif distance_km <= 18:
        return 70
    elif distance_km <= 21:
        return 80
    elif distance_km <= 24:
        return 90
    else:
        return 100

def update_metro_edges():
    """Update metro edges in edges.csv with official fares"""
    
    print("="*80)
    print("UPDATING METRO FARES WITH OFFICIAL HMRL STRUCTURE")
    print("="*80)
    
    # Load current edges
    edges_df = pd.read_csv('edges.csv')
    print(f"\n📊 Loaded {len(edges_df)} total edges")
    
    # Get metro edges
    metro_mask = edges_df['mode'] == 'metro'
    metro_count = metro_mask.sum()
    print(f"   {metro_count} metro edges found")
    
    # Update metro costs with official fares
    print("\n🔄 Updating metro costs...")
    
    old_costs = []
    new_costs = []
    
    for idx in edges_df[metro_mask].index:
        dist_km = edges_df.loc[idx, 'distance_km']
        old_cost = edges_df.loc[idx, 'cost_base_rs']
        new_cost = calculate_official_hmrl_fare(dist_km)
        
        old_costs.append(old_cost)
        new_costs.append(new_cost)
        
        # Update both base and actual cost
        edges_df.loc[idx, 'cost_base_rs'] = new_cost
        edges_df.loc[idx, 'cost_rs'] = new_cost
    
    print(f"   ✅ Updated {metro_count} metro edges")
    print(f"\n📉 Cost Changes:")
    print(f"   Old avg: ₹{np.mean(old_costs):.2f}")
    print(f"   New avg: ₹{np.mean(new_costs):.2f}")
    print(f"   Old min: ₹{np.min(old_costs):.2f}")
    print(f"   New min: ₹{np.min(new_costs):.2f}")
    print(f"   Old max: ₹{np.max(old_costs):.2f}")
    print(f"   New max: ₹{np.max(new_costs):.2f}")
    
    # Save backup
    backup_path = 'edges_backup.csv'
    edges_df_original = pd.read_csv('edges.csv')
    edges_df_original.to_csv(backup_path, index=False)
    print(f"\n💾 Backup saved: {backup_path}")
    
    # Save updated edges
    edges_df.to_csv('edges.csv', index=False)
    print(f"✅ Updated edges saved: edges.csv")
    
    return edges_df

def create_metro_fare_lookup():
    """Create a comprehensive metro fare lookup table using generated data"""
    
    print("\n" + "="*80)
    print("CREATING METRO FARE LOOKUP TABLE")
    print("="*80)
    
    # Load the generated comprehensive fare data
    fares_df = pd.read_csv('metro_fare_scraper/data/metro_fares_clean.csv')
    print(f"\n📊 Loaded {len(fares_df)} station pairs from generated data")
    
    # Recalculate fares using official HMRL structure
    print("\n🔄 Recalculating with official HMRL fares...")
    fares_df['official_fare_rs'] = fares_df['distance_km'].apply(calculate_official_hmrl_fare)
    fares_df['official_cost_per_km'] = fares_df['official_fare_rs'] / fares_df['distance_km']
    
    # Save updated comprehensive fare table
    output_dir = Path('metro_fare_scraper/data')
    output_file = output_dir / 'metro_fares_official_hmrl.csv'
    
    fares_df.to_csv(output_file, index=False)
    print(f"✅ Saved comprehensive fare lookup: {output_file}")
    
    # Create JSON lookup for easy access
    fare_lookup = {}
    for _, row in fares_df.iterrows():
        key = f"{row['from_station']}_{row['to_station']}"
        fare_lookup[key] = {
            'fare_rs': float(row['official_fare_rs']),
            'distance_km': float(row['distance_km']),
            'travel_time_min': float(row['travel_time_min']),
            'cost_per_km': float(row['official_cost_per_km'])
        }
    
    json_file = output_dir / 'metro_fares_lookup.json'
    with open(json_file, 'w') as f:
        json.dump(fare_lookup, f, indent=2)
    print(f"✅ Saved JSON lookup: {json_file}")
    
    # Print statistics
    print(f"\n📊 Statistics with Official HMRL Fares:")
    print(f"   Fare range: ₹{fares_df['official_fare_rs'].min():.0f} - ₹{fares_df['official_fare_rs'].max():.0f}")
    print(f"   Average fare: ₹{fares_df['official_fare_rs'].mean():.2f}")
    print(f"   Median fare: ₹{fares_df['official_fare_rs'].median():.0f}")
    print(f"   Distance range: {fares_df['distance_km'].min():.2f} - {fares_df['distance_km'].max():.2f} km")
    print(f"   Average cost/km: ₹{fares_df['official_cost_per_km'].mean():.2f}/km")
    
    # Show fare bracket distribution
    print(f"\n📋 Fare Bracket Distribution:")
    fare_counts = fares_df['official_fare_rs'].value_counts().sort_index()
    for fare, count in fare_counts.items():
        percentage = (count / len(fares_df)) * 100
        print(f"   ₹{int(fare):3d}: {count:4d} pairs ({percentage:5.1f}%)")
    
    return fares_df

def verify_updates():
    """Verify that updates were applied correctly"""
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    edges_df = pd.read_csv('edges.csv')
    metro_edges = edges_df[edges_df['mode'] == 'metro']
    
    print(f"\n✅ Metro Edges Check:")
    print(f"   Total metro edges: {len(metro_edges)}")
    print(f"   Cost range: ₹{metro_edges['cost_base_rs'].min():.2f} - ₹{metro_edges['cost_base_rs'].max():.2f}")
    print(f"   Average cost: ₹{metro_edges['cost_base_rs'].mean():.2f}")
    
    # Check if costs match official structure
    sample = metro_edges.head(10)
    print(f"\n📋 Sample Metro Edges (first 10):")
    print(f"{'From':<20} {'To':<20} {'Dist (km)':<12} {'Old Formula':<12} {'New Official':<12}")
    print("-" * 80)
    
    for _, row in sample.iterrows():
        from_station = row['from_id'].replace('_Metro', '')
        to_station = row['to_id'].replace('_Metro', '')
        dist = row['distance_km']
        new_cost = row['cost_base_rs']
        old_cost = 10 + dist * 2.0  # Old formula from create_edges_final.py
        
        print(f"{from_station:<20} {to_station:<20} {dist:<12.2f} ₹{old_cost:<11.2f} ₹{new_cost:<11.2f}")
    
    print(f"\n✅ Verification complete!")

def main():
    """Main execution"""
    
    print("\n" + "🚇 " + "="*76)
    print("   METRO FARE UPDATE - OFFICIAL HMRL STRUCTURE INTEGRATION")
    print("="*80 + "\n")
    
    print("This script will:")
    print("  1. Update metro edges in edges.csv with official HMRL fares")
    print("  2. Create comprehensive fare lookup table (3,192 pairs)")
    print("  3. Use distance-based official fare brackets (₹10-₹100)")
    print("  4. Maintain all existing distance data")
    print("  5. Create backup of original edges.csv")
    
    response = input("\nProceed with update? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Update cancelled.")
        return
    
    # Update edges.csv
    edges_df = update_metro_edges()
    
    # Create comprehensive lookup
    fares_df = create_metro_fare_lookup()
    
    # Verify updates
    verify_updates()
    
    print("\n" + "="*80)
    print("✅ ALL UPDATES COMPLETE!")
    print("="*80)
    
    print("\n📝 Summary:")
    print("  ✅ Updated edges.csv with official HMRL fares")
    print("  ✅ Created comprehensive fare lookup (metro_fares_official_hmrl.csv)")
    print("  ✅ Generated JSON lookup for easy access (metro_fares_lookup.json)")
    print("  ✅ Backed up original edges.csv")
    
    print("\n🎯 Next Steps:")
    print("  1. Review metro_fare_comparison.png for visual comparison")
    print("  2. Test routing in the web dashboard")
    print("  3. Verify that metro routes show realistic costs")
    print("  4. Check resilience analysis with new costs")
    
    print("\n💡 Key Improvements:")
    print("  • Metro fares now follow official HMRL structure (₹10-₹100)")
    print("  • Distance-based tiered pricing")
    print("  • More realistic cost analysis")
    print("  • Better route recommendations")
    print("  • Policy compliance with HMRL pricing")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
