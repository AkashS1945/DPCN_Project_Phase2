# Hyderabad Metro Fare & Distance Scraper

This project scrapes official fare and distance data from the L&T Metro Rail (Hyderabad) website.

## 🎯 Objective

Extract complete fare matrix and inter-station distances for all Hyderabad Metro stations from the official website: https://ltmetro.com/find-trip-details/

## 📁 Project Structure

```
metro_fare_scraper/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scrape_metro_data.py              # Main scraper script (Selenium)
├── scrape_metro_api.py               # Alternative API-based scraper
├── analyze_fares.py                  # Analysis & visualization script
├── data/                             # Output data folder
│   ├── metro_fares_raw.csv           # Raw scraped data
│   ├── metro_fares_clean.csv         # Cleaned data
│   └── metro_fares_matrix.json       # Fare matrix format
└── visualizations/                   # Output charts
    ├── fare_heatmap.png
    ├── distance_distribution.png
    └── fare_analysis.png
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Scraper

```bash
# Option 1: Using Selenium (recommended)
python scrape_metro_data.py

# Option 2: Using API endpoint (if available)
python scrape_metro_api.py
```

### 3. Analyze Results

```bash
python analyze_fares.py
```

## 📊 Expected Output

- **metro_fares_raw.csv**: All station pairs with fare and distance
- **Fare statistics**: Min, max, average fares
- **Distance analysis**: Inter-station distances
- **Visualizations**: Heatmaps and distribution charts

## 🔧 Requirements

- Python 3.8+
- Chrome/Chromium browser (for Selenium)
- Internet connection

## 📝 Notes

- Scraping respects rate limits (2-3 seconds delay between requests)
- Total combinations: ~3,250 station pairs (57 stations × 57 stations)
- Estimated scraping time: 2-3 hours for complete data
- Data is cached to avoid re-scraping

## 🎓 Metro Lines Covered

- **Red Line**: Miyapur ↔ LB Nagar (27 stations)
- **Blue Line**: Nagole ↔ Raidurg (13 stations)  
- **Green Line**: JBS Parade Ground ↔ Falaknuma (17 stations)

## 📈 Analysis Features

1. Fare distribution analysis
2. Distance vs Fare correlation
3. Line-wise statistics
4. Interchange station analysis
5. Cost-per-kilometer analysis

## ⚠️ Legal Notice

This scraper is for educational and research purposes only. Please respect the website's terms of service and robots.txt.
