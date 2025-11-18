"""
Hyderabad Metro Fare & Distance Scraper
========================================

This script scrapes official fare and distance data from ltmetro.com
for all station pairs in the Hyderabad Metro network.

Author: DPCN Project
Date: November 2025
"""

import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import os


class MetroFareScraper:
    """Scraper for Hyderabad Metro fare and distance data"""
    
    def __init__(self, headless=True):
        """Initialize the scraper with Chrome options"""
        self.url = "https://ltmetro.com/find-trip-details/"
        self.data = []
        self.stations = []
        self.headless = headless
        self.driver = None
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
    def setup_driver(self):
        """Setup Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Set binary location for Chromium snap
        chrome_options.binary_location = '/snap/bin/chromium'
        
        # Try to auto-install ChromeDriver, fallback to system driver
        try:
            service = Service(ChromeDriverManager().install())
        except:
            # Use system chromedriver if webdriver-manager fails
            service = None
        
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("✅ Chrome WebDriver initialized successfully")
        
    def get_stations(self):
        """Extract all station names from the dropdown"""
        print(f"\n🔍 Loading page: {self.url}")
        self.driver.get(self.url)
        
        # Wait for page to load
        time.sleep(3)
        
        try:
            # Wait for the "From" dropdown to be available
            wait = WebDriverWait(self.driver, 15)
            from_dropdown = wait.until(
                EC.presence_of_element_located((By.ID, "from_station"))
            )
            
            # Get all station options
            from_select = Select(from_dropdown)
            self.stations = [
                option.text.strip() 
                for option in from_select.options 
                if option.text.strip() and option.value
            ]
            
            print(f"✅ Found {len(self.stations)} metro stations:")
            for i, station in enumerate(self.stations, 1):
                print(f"   {i:2d}. {station}")
            
            return self.stations
            
        except Exception as e:
            print(f"❌ Error extracting stations: {str(e)}")
            return []
    
    def scrape_station_pair(self, from_station, to_station):
        """Scrape fare and distance for a specific station pair"""
        try:
            # Select From station
            from_select = Select(self.driver.find_element(By.ID, "from_station"))
            from_select.select_by_visible_text(from_station)
            time.sleep(0.5)
            
            # Select To station
            to_select = Select(self.driver.find_element(By.ID, "to_station"))
            to_select.select_by_visible_text(to_station)
            time.sleep(0.5)
            
            # Click search/submit button
            search_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
            search_btn.click()
            time.sleep(2)
            
            # Wait for results to load
            wait = WebDriverWait(self.driver, 10)
            
            # Try multiple possible selectors for fare and distance
            fare = None
            distance = None
            travel_time = None
            
            # Extract fare
            try:
                fare_selectors = [
                    ".fare-amount", ".trip-fare", ".fare", 
                    "//div[contains(text(), '₹')]",
                    "//span[contains(@class, 'fare')]",
                    "//*[contains(text(), 'Fare')]/following-sibling::*"
                ]
                
                for selector in fare_selectors:
                    try:
                        if selector.startswith("//"):
                            element = self.driver.find_element(By.XPATH, selector)
                        else:
                            element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        fare = element.text.strip()
                        if fare:
                            break
                    except:
                        continue
                        
            except Exception as e:
                print(f"   ⚠️  Could not extract fare: {str(e)}")
            
            # Extract distance
            try:
                distance_selectors = [
                    ".distance", ".trip-distance", ".route-distance",
                    "//div[contains(text(), 'km')]",
                    "//span[contains(@class, 'distance')]",
                    "//*[contains(text(), 'Distance')]/following-sibling::*"
                ]
                
                for selector in distance_selectors:
                    try:
                        if selector.startswith("//"):
                            element = self.driver.find_element(By.XPATH, selector)
                        else:
                            element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        distance = element.text.strip()
                        if distance:
                            break
                    except:
                        continue
                        
            except Exception as e:
                print(f"   ⚠️  Could not extract distance: {str(e)}")
            
            # Extract travel time (if available)
            try:
                time_selectors = [
                    ".travel-time", ".trip-time", ".duration",
                    "//div[contains(text(), 'min')]",
                    "//*[contains(text(), 'Time')]/following-sibling::*"
                ]
                
                for selector in time_selectors:
                    try:
                        if selector.startswith("//"):
                            element = self.driver.find_element(By.XPATH, selector)
                        else:
                            element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        travel_time = element.text.strip()
                        if travel_time:
                            break
                    except:
                        continue
            except:
                pass
            
            return {
                'from_station': from_station,
                'to_station': to_station,
                'fare': fare,
                'distance': distance,
                'travel_time': travel_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None
    
    def scrape_all_pairs(self, sample_size=None):
        """Scrape all station pairs (or a sample)"""
        if not self.stations:
            print("❌ No stations found. Run get_stations() first.")
            return
        
        total_pairs = len(self.stations) * (len(self.stations) - 1)
        
        if sample_size:
            print(f"\n📊 Scraping {sample_size} sample station pairs (out of {total_pairs} total)")
        else:
            print(f"\n📊 Scraping ALL {total_pairs} station pairs")
            print(f"⏱️  Estimated time: {total_pairs * 3 / 60:.1f} minutes")
        
        processed = 0
        errors = 0
        
        for i, from_station in enumerate(self.stations, 1):
            for j, to_station in enumerate(self.stations, 1):
                # Skip same station
                if from_station == to_station:
                    continue
                
                # Check sample size limit
                if sample_size and processed >= sample_size:
                    print(f"\n✅ Sample limit reached ({sample_size} pairs)")
                    return
                
                processed += 1
                
                print(f"\n[{processed}/{total_pairs if not sample_size else sample_size}] {from_station} → {to_station}")
                
                result = self.scrape_station_pair(from_station, to_station)
                
                if result and result['fare'] and result['distance']:
                    self.data.append(result)
                    print(f"   ✅ Fare: {result['fare']}, Distance: {result['distance']}")
                else:
                    errors += 1
                    print(f"   ⚠️  Failed to extract data")
                
                # Go back to search page
                self.driver.get(self.url)
                time.sleep(2)
                
                # Save checkpoint every 50 records
                if processed % 50 == 0:
                    self.save_checkpoint()
        
        print(f"\n✅ Scraping complete!")
        print(f"   Total pairs processed: {processed}")
        print(f"   Successful: {len(self.data)}")
        print(f"   Errors: {errors}")
    
    def save_checkpoint(self):
        """Save current data as checkpoint"""
        if self.data:
            checkpoint_file = f"data/checkpoint_{len(self.data)}_records.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"   💾 Checkpoint saved: {checkpoint_file}")
    
    def save_data(self):
        """Save scraped data to CSV and JSON"""
        if not self.data:
            print("❌ No data to save")
            return
        
        # Save as CSV
        df = pd.DataFrame(self.data)
        csv_file = 'data/metro_fares_raw.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Data saved to {csv_file}")
        
        # Save as JSON
        json_file = 'data/metro_fares_raw.json'
        with open(json_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"✅ Data saved to {json_file}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Total records: {len(df)}")
        print(f"   Unique stations: {df['from_station'].nunique()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("\n✅ Browser closed")


def main():
    """Main execution function"""
    print("="*70)
    print("  HYDERABAD METRO FARE & DISTANCE SCRAPER")
    print("="*70)
    
    # Initialize scraper
    scraper = MetroFareScraper(headless=False)  # Set to True for background mode
    
    try:
        # Setup driver
        scraper.setup_driver()
        
        # Get all stations
        stations = scraper.get_stations()
        
        if not stations:
            print("❌ Failed to extract stations. Please check the website.")
            return
        
        # Ask user for scraping mode
        print("\n" + "="*70)
        print("Choose scraping mode:")
        print("  1. Sample (10 pairs) - Quick test")
        print("  2. Small (100 pairs) - Medium test")
        print("  3. Full scrape (All pairs) - Complete dataset")
        print("="*70)
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            scraper.scrape_all_pairs(sample_size=10)
        elif choice == '2':
            scraper.scrape_all_pairs(sample_size=100)
        elif choice == '3':
            confirm = input(f"\n⚠️  This will scrape {len(stations) * (len(stations) - 1)} pairs. Continue? (yes/no): ")
            if confirm.lower() == 'yes':
                scraper.scrape_all_pairs()
            else:
                print("Cancelled.")
                return
        else:
            print("Invalid choice. Running sample mode.")
            scraper.scrape_all_pairs(sample_size=10)
        
        # Save data
        df = scraper.save_data()
        
        if df is not None:
            print("\n" + "="*70)
            print("SAMPLE DATA (First 5 records):")
            print("="*70)
            print(df.head())
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always close the browser
        scraper.close()


if __name__ == "__main__":
    main()
