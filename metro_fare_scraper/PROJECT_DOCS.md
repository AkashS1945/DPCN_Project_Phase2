# Metro Fare Scraper - Project Documentation

## 📁 Complete File Structure

```
metro_fare_scraper/
│
├── README.md                      # Main project documentation
├── QUICKSTART.md                  # Quick start guide
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── setup.sh                       # Setup script (Linux/Mac)
├── test_setup.py                  # System verification script
│
├── scrape_metro_data.py          # Main scraper (Selenium-based)
├── scrape_metro_api.py           # Alternative API scraper
├── analyze_fares.py              # Data analysis & visualization
│
├── data/                         # Output data folder
│   ├── .gitkeep
│   ├── metro_fares_raw.csv       # Raw scraped data (generated)
│   ├── metro_fares_clean.csv     # Cleaned data (generated)
│   ├── fare_statistics.json      # Statistics (generated)
│   ├── station_list.txt          # Station reference (generated)
│   └── checkpoint_*.json         # Scraping checkpoints (generated)
│
└── visualizations/               # Charts and reports
    ├── .gitkeep
    ├── fare_distribution.png     # Generated charts
    ├── distance_distribution.png
    ├── fare_vs_distance.png
    ├── cost_per_km_analysis.png
    ├── fare_heatmap_interactive.html
    ├── analysis_dashboard.html
    └── analysis_report.txt
```

## 🎯 Project Objectives

1. **Data Collection**: Scrape official fare and distance data from ltmetro.com
2. **Data Processing**: Clean and standardize the scraped data
3. **Analysis**: Generate statistical insights and correlations
4. **Visualization**: Create charts, heatmaps, and interactive dashboards
5. **Documentation**: Provide comprehensive usage guides

## 🚀 Complete Workflow

### Phase 1: Setup (5 minutes)

```bash
cd metro_fare_scraper/

# Option A: Automated setup
./setup.sh

# Option B: Manual setup
pip install -r requirements.txt
mkdir -p data visualizations

# Verify setup
python test_setup.py
```

**Expected Output:**
```
✅ All required packages are installed!
✅ Chrome/Chromium found
✅ Website accessible
✅ All checks passed!
```

### Phase 2: Data Scraping (2-3 hours or 5 minutes for sample)

```bash
python scrape_metro_data.py
```

**Interactive Options:**
```
Choose scraping mode:
  1. Sample (10 pairs) - Quick test
  2. Small (100 pairs) - Medium test
  3. Full scrape (All pairs) - Complete dataset
```

**What Happens:**
1. Launches Chrome browser (automated)
2. Navigates to ltmetro.com/find-trip-details/
3. Extracts all station names from dropdown
4. Iterates through station pairs
5. For each pair:
   - Selects From station
   - Selects To station
   - Clicks Search
   - Extracts fare, distance, travel time
   - Saves to data structure
6. Saves checkpoint every 50 records
7. Generates final CSV and JSON files

**Output Files:**
- `data/metro_fares_raw.csv` - Complete dataset
- `data/metro_fares_raw.json` - JSON format
- `data/checkpoint_*.json` - Recovery checkpoints

### Phase 3: Data Analysis (1 minute)

```bash
python analyze_fares.py
```

**Processing Steps:**
1. Loads raw data
2. Cleans and standardizes:
   - Extracts numeric fare from "₹20"
   - Extracts distance from "5.2 km"
   - Calculates cost per kilometer
3. Generates statistics
4. Creates visualizations
5. Produces comprehensive report

**Generated Files:**
- `data/metro_fares_clean.csv`
- `data/fare_statistics.json`
- 6 visualization files (PNG & HTML)
- `visualizations/analysis_report.txt`

## 📊 Data Schema

### Raw Data (metro_fares_raw.csv)

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| from_station | String | "Miyapur" | Origin station name |
| to_station | String | "LB Nagar" | Destination station |
| fare | String | "₹50" | Fare with currency symbol |
| distance | String | "29.4 km" | Distance with unit |
| travel_time | String | "58 min" | Estimated travel time |
| timestamp | ISO String | "2025-11-16T10:30:00" | Scrape timestamp |

### Cleaned Data (metro_fares_clean.csv)

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| from_station | String | "Miyapur" | Origin station |
| to_station | String | "LB Nagar" | Destination station |
| fare_rs | Float | 50.0 | Fare in rupees |
| distance_km | Float | 29.4 | Distance in kilometers |
| travel_time_min | Float | 58.0 | Time in minutes |
| cost_per_km | Float | 1.70 | Cost efficiency |

## 📈 Analysis Features

### 1. Statistical Analysis

```python
{
  "fare": {
    "min": 10,
    "max": 50,
    "mean": 28.5,
    "median": 30,
    "std": 12.3
  },
  "distance": {
    "min": 1.2,
    "max": 42.8,
    "mean": 15.2,
    "median": 14.5
  }
}
```

### 2. Visualizations

**Static Charts (PNG):**
- Fare distribution histogram & box plot
- Distance distribution & cumulative plot
- Fare vs Distance scatter with regression
- Cost-per-km analysis

**Interactive Dashboards (HTML):**
- Fare heatmap (all station pairs)
- Multi-panel analysis dashboard
- Zoomable, filterable visualizations

### 3. Reports

Text-based comprehensive report including:
- Dataset summary
- Fare & distance statistics
- Top 5 most expensive routes
- Top 5 longest routes
- Cost efficiency analysis

## 🔧 Technical Details

### Scraper Architecture (scrape_metro_data.py)

```python
class MetroFareScraper:
    - setup_driver()          # Initialize Selenium WebDriver
    - get_stations()          # Extract station list
    - scrape_station_pair()   # Scrape single pair
    - scrape_all_pairs()      # Iterate all combinations
    - save_checkpoint()       # Save progress
    - save_data()            # Export final data
```

**Key Technologies:**
- **Selenium**: Browser automation
- **WebDriver Manager**: Auto-install ChromeDriver
- **Pandas**: Data manipulation
- **BeautifulSoup**: HTML parsing (backup)

### Analyzer Architecture (analyze_fares.py)

```python
class MetroFareAnalyzer:
    - load_data()             # Read CSV
    - clean_data()            # Standardize format
    - calculate_statistics()  # Compute metrics
    - plot_*()               # Generate charts
    - create_dashboard()     # Interactive viz
    - generate_report()      # Text report
```

**Key Libraries:**
- **Pandas**: Data processing
- **Matplotlib**: Static charts
- **Seaborn**: Statistical plots
- **Plotly**: Interactive visualizations

## 🎓 Hyderabad Metro Network Details

### Lines & Stations

**Red Line (Corridor 1):** 27 stations
- Route: Miyapur ↔ LB Nagar
- Length: ~29 km
- Color: Red (#E31E24)

**Blue Line (Corridor 3):** 22 stations
- Route: Nagole ↔ Raidurg
- Length: ~27 km
- Color: Blue (#007ABB)

**Green Line (Corridor 2):** 15 stations
- Route: JBS Parade Ground ↔ Falaknuma
- Length: ~14 km (under expansion)
- Color: Green (#009846)

### Interchange Stations

1. **Ameerpet** - Red ↔ Blue
2. **MG Bus Station** - Red ↔ Green
3. **Parade Ground** - Blue ↔ Green (future)

### Fare Structure (Official)

| Distance (km) | Fare (₹) |
|--------------|---------|
| 0 - 2 | 10 |
| 2 - 4 | 15 |
| 4 - 6 | 20 |
| 6 - 9 | 25 |
| 9 - 12 | 30 |
| 12 - 15 | 35 |
| 15 - 20 | 40 |
| 20 - 25 | 45 |
| 25+ | 50 |

## 🔍 Troubleshooting Guide

### Problem: Selenium not found

```bash
pip install selenium webdriver-manager
```

### Problem: ChromeDriver error

```bash
# Auto-install with webdriver-manager (already in code)
# Or manual install:
sudo apt install chromium-chromedriver
```

### Problem: Scraping returns no data

**Solutions:**
1. Check internet connection
2. Verify website is accessible: https://ltmetro.com/find-trip-details/
3. Run in non-headless mode (`headless=False`)
4. Check for website structure changes
5. Inspect element IDs in browser DevTools

### Problem: Analysis fails

```bash
# Check if data file exists
ls -lh data/metro_fares_raw.csv

# Verify CSV structure
head data/metro_fares_raw.csv

# Re-run with verbose output
python analyze_fares.py
```

### Problem: Missing visualizations

```bash
# Install plotting libraries
pip install matplotlib seaborn plotly

# Check directory permissions
chmod 755 visualizations/
```

## 📚 Use Cases

### 1. Research & Analysis
- Study fare-distance correlation
- Identify pricing patterns
- Compare with international metros

### 2. Integration with Network Model
```python
import pandas as pd

# Load metro fares
metro_fares = pd.read_csv('data/metro_fares_clean.csv')

# Merge with network edges
edges = pd.read_csv('../edges.csv')
metro_edges = edges[edges['mode'] == 'metro']
merged = metro_edges.merge(metro_fares, 
                           left_on=['from', 'to'],
                           right_on=['from_station', 'to_station'])
```

### 3. Route Cost Optimization
- Calculate total trip cost
- Find cheapest routes
- Multi-modal cost comparison

### 4. Accessibility Studies
- Fare affordability analysis
- Distance-based equity
- Station connectivity costs

## 🔐 Legal & Ethical Considerations

1. **Educational Use**: This scraper is for research and education
2. **Rate Limiting**: Respects 2-3 second delays between requests
3. **Data Accuracy**: Always verify against official sources
4. **Terms of Service**: Review ltmetro.com ToS before large-scale scraping
5. **Attribution**: Credit L&T Metro Rail Hyderabad for data

## 🚀 Future Enhancements

### Planned Features
- [ ] Peak/off-peak fare tracking
- [ ] Multi-trip discount analysis
- [ ] QR code/token fare comparison
- [ ] Line-wise statistics
- [ ] Temporal fare changes tracking
- [ ] API endpoint integration

### Integration Possibilities
- Import into PostgreSQL/SQLite database
- Build REST API for fare queries
- Create mobile app for fare calculator
- Generate route cost comparisons

## 📞 Support & Contribution

### Getting Help
1. Check `QUICKSTART.md` for common issues
2. Run `python test_setup.py` to diagnose
3. Review console error messages
4. Check `data/checkpoint_*.json` for partial data

### Contributing
Improvements welcome:
- Better error handling
- Additional visualizations
- Performance optimizations
- Multi-city support

## 📄 License

This project is for educational purposes. Data belongs to L&T Metro Rail Hyderabad Limited.

---

**Project Status:** ✅ Ready for Use

**Last Updated:** November 2025

**Total Lines of Code:** ~1,200

**Estimated Setup Time:** 5 minutes

**Estimated Scrape Time:** 2-3 hours (full) or 2 minutes (sample)

**Estimated Analysis Time:** 1 minute

---

*Happy Scraping and Analysis! 🚇📊*
