"""
Alternative Metro Fare Scraper using API endpoint
==================================================

This script attempts to use the website's backend API directly
instead of browser automation (faster and more reliable if available).
"""

import requests
import json
import pandas as pd
import time
from datetime import datetime
import os


class MetroAPIFareScraper:
    """API-based scraper for metro fares"""
    
    def __init__(self):
        self.base_url = "https://ltmetro.com"
        self.api_endpoint = None  # To be discovered
        self.session = requests.Session()
        self.data = []
        
        # Setup headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://ltmetro.com/find-trip-details/'
        }
        
        os.makedirs('data', exist_ok=True)
    
    def discover_api_endpoint(self):
        """
        Try to discover the API endpoint used by the website.
        
        Instructions for manual discovery:
        1. Open https://ltmetro.com/find-trip-details/ in Chrome
        2. Open DevTools (F12) → Network tab
        3. Select a from/to station and click search
        4. Look for XHR/Fetch requests in the Network tab
        5. Find the request that returns fare/distance data
        6. Copy the request URL and payload format
        """
        
        print("🔍 API Endpoint Discovery")
        print("="*70)
        print("\nTo find the API endpoint manually:")
        print("1. Open: https://ltmetro.com/find-trip-details/")
        print("2. Open Chrome DevTools (F12) → Network tab")
        print("3. Filter by 'XHR' or 'Fetch'")
        print("4. Select stations and submit")
        print("5. Look for API call with fare/distance data")
        print("\nCommon patterns to look for:")
        print("  - /wp-admin/admin-ajax.php")
        print("  - /api/trip-details")
        print("  - /get-fare")
        print("="*70)
        
        # Common WordPress AJAX endpoints
        possible_endpoints = [
            f"{self.base_url}/wp-admin/admin-ajax.php",
            f"{self.base_url}/wp-json/metro/v1/trip-details",
            f"{self.base_url}/api/get-trip-details",
            f"{self.base_url}/ajax/fare-calculator"
        ]
        
        print("\n🔍 Testing common API endpoints...")
        for endpoint in possible_endpoints:
            print(f"   Testing: {endpoint}")
            try:
                response = self.session.get(endpoint, headers=self.headers, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ Endpoint accessible: {endpoint}")
                    self.api_endpoint = endpoint
                    return endpoint
            except:
                pass
        
        print("\n⚠️  Could not auto-discover API endpoint")
        print("   Please check browser DevTools and update the script manually")
        return None
    
    def get_fare_data(self, from_station, to_station):
        """
        Fetch fare data using API endpoint.
        
        This is a template - adjust based on actual API structure.
        """
        if not self.api_endpoint:
            print("❌ API endpoint not configured")
            return None
        
        # Example payload (adjust based on actual API)
        payload = {
            'action': 'get_trip_details',  # Common WordPress AJAX action
            'from_station': from_station,
            'to_station': to_station,
            'nonce': ''  # May require nonce/token from page
        }
        
        try:
            response = self.session.post(
                self.api_endpoint,
                data=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"   ❌ API returned status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ API request failed: {str(e)}")
            return None
    
    def manual_data_entry(self):
        """
        Helper to manually enter data if API scraping fails.
        Reads from a manually created CSV file.
        """
        print("\n📝 Manual Data Entry Mode")
        print("="*70)
        print("Since automated scraping may not work, you can:")
        print("1. Manually collect fare data from the website")
        print("2. Save in: data/manual_fares.csv")
        print("3. Format: from_station,to_station,fare,distance")
        print("="*70)
        
        manual_file = 'data/manual_fares.csv'
        if os.path.exists(manual_file):
            df = pd.read_csv(manual_file)
            print(f"\n✅ Loaded {len(df)} records from {manual_file}")
            return df
        else:
            print(f"\n⚠️  File not found: {manual_file}")
            print("   Create this file with your manual data")
            return None


def create_fare_matrix_template():
    """
    Create a template showing Metro stations for manual data entry.
    """
    
    # Known Hyderabad Metro stations (as of 2025)
    metro_stations = {
        'Red Line': [
            'Miyapur', 'JNTU College', 'KPHB Colony', 'Kukatpally', 'Balanagar',
            'Moosapet', 'Bharath Nagar', 'Erragadda', 'ESI Hospital', 'SR Nagar',
            'Ameerpet', 'Punjagutta', 'Irrum Manzil', 'Khairatabad', 'Lakdi Ka Pul',
            'Assembly', 'Nampally', 'Gandhi Bhavan', 'Osmania Medical College',
            'MG Bus Station', 'Malakpet', 'New Market', 'Musarambagh',
            'Dilsukhnagar', 'Chaitanyapuri', 'Victoria Memorial', 'LB Nagar'
        ],
        'Blue Line': [
            'Nagole', 'Uppal', 'Survey of India', 'NGRI', 'Habsiguda',
            'Tarnaka', 'Mettuguda', 'Secunderabad East', 'Paradise',
            'Rasoolpura', 'Prakash Nagar', 'Begumpet', 'Ameerpet', 'Madhura Nagar',
            'Yousufguda', 'Road No. 5 Jubilee Hills', 'Jubilee Hills Check Post',
            'Peddamma Gudi', 'Madhapur', 'Durgam Cheruvu', 'HITEC City',
            'Raidurg'
        ],
        'Green Line': [
            'JBS Parade Ground', 'Secunderabad West', 'Gandhi Hospital',
            'Musheerabad', 'RTC X Roads', 'Chikkadpally', 'Narayanguda',
            'Sultan Bazar', 'MG Bus Station', 'Malakpet', 'New Market',
            'Musarambagh', 'Dilsukhnagar', 'Moosrambagh', 'Nagole'
        ]
    }
    
    # Create template CSV
    template_file = 'data/station_list.txt'
    with open(template_file, 'w') as f:
        f.write("Hyderabad Metro Stations\n")
        f.write("="*70 + "\n\n")
        
        for line, stations in metro_stations.items():
            f.write(f"\n{line} ({len(stations)} stations):\n")
            f.write("-" * 50 + "\n")
            for i, station in enumerate(stations, 1):
                f.write(f"{i:2d}. {station}\n")
    
    print(f"✅ Station list saved to: {template_file}")
    
    # Create empty template for manual entry
    template_csv = 'data/manual_fares_template.csv'
    df_template = pd.DataFrame(columns=[
        'from_station', 'to_station', 'fare_rs', 'distance_km', 'line'
    ])
    df_template.to_csv(template_csv, index=False)
    print(f"✅ Template CSV created: {template_csv}")
    
    return metro_stations


def main():
    """Main execution"""
    print("="*70)
    print("  METRO FARE API SCRAPER")
    print("="*70)
    
    scraper = MetroAPIFareScraper()
    
    # Try to discover API
    endpoint = scraper.discover_api_endpoint()
    
    if not endpoint:
        print("\n⚠️  API-based scraping not available")
        print("\nRecommended alternatives:")
        print("  1. Use scrape_metro_data.py (Selenium-based)")
        print("  2. Manual data collection")
        print("  3. Use existing GTFS data with distance calculations")
    
    # Create station list and template
    create_fare_matrix_template()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python scrape_metro_data.py (for automated scraping)")
    print("2. Or manually fill: data/manual_fares_template.csv")
    print("3. Then run: python analyze_fares.py")
    print("="*70)


if __name__ == "__main__":
    main()
