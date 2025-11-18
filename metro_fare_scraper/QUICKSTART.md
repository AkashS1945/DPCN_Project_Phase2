# Metro Fare Scraper - Quick Start Guide

## 🎯 Purpose

This project scrapes and analyzes official fare and distance data for all station pairs in the Hyderabad Metro network from https://ltmetro.com/find-trip-details/

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
mkdir -p data visualizations
```

### Step 2: Scrape Data

```bash
python scrape_metro_data.py
```

**What it does:**
- Opens ltmetro.com in automated browser
- Extracts all 57 metro stations
- Tests all station pair combinations (~3,250 pairs)
- Scrapes fare, distance, and travel time
- Saves to `data/metro_fares_raw.csv`

**Time estimate:** 2-3 hours for complete scrape (or choose sample mode for quick test)

### Step 3: Analyze Results

```bash
python analyze_fares.py
```

**Generates:**
- Cleaned dataset: `data/metro_fares_clean.csv`
- Statistics: `data/fare_statistics.json`
- Charts: `visualizations/*.png`
- Interactive dashboards: `visualizations/*.html`
- Text report: `visualizations/analysis_report.txt`

## 📊 Expected Outputs

### Data Files

1. **metro_fares_raw.csv** - Raw scraped data
   ```csv
   from_station,to_station,fare,distance,travel_time,timestamp
   Miyapur,LB Nagar,₹50,29.4 km,58 min,2025-11-16T10:30:00
   ```

2. **metro_fares_clean.csv** - Cleaned and processed
   ```csv
   from_station,to_station,fare_rs,distance_km,cost_per_km
   Miyapur,LB Nagar,50,29.4,1.70
   ```

### Visualizations

1. **fare_distribution.png** - Histogram and box plot of fares
2. **distance_distribution.png** - Distance analysis
3. **fare_vs_distance.png** - Correlation scatter plot
4. **cost_per_km_analysis.png** - Cost efficiency analysis
5. **fare_heatmap_interactive.html** - Interactive fare matrix
6. **analysis_dashboard.html** - Comprehensive dashboard

### Statistics

Example output:
```
Total Station Pairs: 3,192
Unique Stations: 57
Fare Range: ₹10 - ₹50
Average Fare: ₹28.50
Average Distance: 15.2 km
Cost per km: ₹1.87/km
```

## 🔧 Troubleshooting

### Issue: Chrome/ChromeDriver not found
```bash
# Ubuntu/Debian
sudo apt install chromium-browser

# Or let script auto-download ChromeDriver
# (it will download automatically on first run)
```

### Issue: Selenium import error
```bash
pip install selenium webdriver-manager
```

### Issue: Website changed structure

If scraping fails:
1. Open DevTools (F12) on the website
2. Check Network tab for API endpoints
3. Update selectors in `scrape_metro_data.py`
4. Or use manual data entry mode

### Issue: Slow scraping

Options:
1. Use sample mode (10-100 pairs for testing)
2. Run in headless mode (set `headless=True`)
3. Adjust sleep timers in code

## 📈 Analysis Features

The analyzer provides:

1. **Statistical Summary**
   - Min, max, mean, median, std deviation
   - For fares, distances, cost-per-km

2. **Distribution Analysis**
   - Fare distribution patterns
   - Distance patterns
   - Cost efficiency trends

3. **Correlation Analysis**
   - Fare vs Distance relationship
   - R² coefficient
   - Regression line

4. **Route Analysis**
   - Most expensive routes
   - Longest routes
   - Best value routes

5. **Interactive Visualizations**
   - Zoomable heatmaps
   - Filterable dashboards
   - Hover tooltips

## 🎓 Metro Network Info

**Lines Covered:**
- Red Line: 27 stations (Miyapur ↔ LB Nagar)
- Blue Line: 22 stations (Nagole ↔ Raidurg)
- Green Line: 15 stations (JBS ↔ Falaknuma - under construction)

**Interchange Stations:**
- Ameerpet (Red ↔ Blue)
- MG Bus Station (Red ↔ Green)
- Parade Ground (Blue ↔ Green)

## 📝 Manual Data Entry (Alternative)

If automated scraping doesn't work:

1. Create `data/manual_fares.csv`:
   ```csv
   from_station,to_station,fare_rs,distance_km
   Miyapur,Ameerpet,20,10.5
   Ameerpet,LB Nagar,35,18.9
   ```

2. Run analyzer on manual data:
   ```bash
   python analyze_fares.py
   ```

## 🔬 Advanced Usage

### Scrape specific stations only

Edit `scrape_metro_data.py`:
```python
# Filter stations
stations = [s for s in stations if s in ['Ameerpet', 'LB Nagar', ...]]
```

### Custom analysis

Edit `analyze_fares.py` to add:
- Line-wise fare comparison
- Peak vs off-peak analysis
- Interchange impact analysis

### Export formats

Add to analyzer:
```python
df.to_excel('data/fares.xlsx')
df.to_json('data/fares.json', orient='records')
```

## ⚠️ Important Notes

1. **Respect rate limits** - Default: 2-3 seconds between requests
2. **Data accuracy** - Verify against official sources
3. **Website changes** - May require selector updates
4. **Legal compliance** - For educational/research use only

## 📞 Support

For issues:
1. Check console output for error messages
2. Review `data/checkpoint_*.json` for partial data
3. Try sample mode first
4. Check website availability

## 🎯 Project Goals

✅ Extract complete fare matrix
✅ Analyze fare-distance relationship  
✅ Identify pricing patterns
✅ Generate visualizations
✅ Create reusable dataset for research

## 📚 Next Steps

After scraping:
1. Integrate with existing transport network data
2. Build route optimization algorithms
3. Compare with bus/auto fare structures
4. Create multimodal cost analysis

---

**Happy Scraping! 🚇📊**
