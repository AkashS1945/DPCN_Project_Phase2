#!/bin/bash

# Setup script for Metro Fare Scraper
# Run this to install dependencies and prepare the environment

echo "=================================================="
echo "  Hyderabad Metro Fare Scraper - Setup"
echo "=================================================="

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "⚠️  Some packages may have failed to install."
    echo "   Try installing individually or check your Python environment."
else
    echo "✅ All dependencies installed successfully!"
fi

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p visualizations

echo "✅ Directories created:"
echo "   - data/ (for CSV/JSON outputs)"
echo "   - visualizations/ (for charts and reports)"

# Check Chrome/Chromium installation
echo ""
echo "🔍 Checking for Chrome/Chromium..."

if command -v google-chrome &> /dev/null; then
    echo "✅ Google Chrome found"
    google-chrome --version
elif command -v chromium-browser &> /dev/null; then
    echo "✅ Chromium found"
    chromium-browser --version
elif command -v chromium &> /dev/null; then
    echo "✅ Chromium found"
    chromium --version
else
    echo "⚠️  Chrome/Chromium not found!"
    echo "   Install Chrome or Chromium for web scraping:"
    echo "   Ubuntu/Debian: sudo apt install chromium-browser"
    echo "   Fedora: sudo dnf install chromium"
    echo "   Or download Chrome: https://www.google.com/chrome/"
fi

echo ""
echo "=================================================="
echo "  SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "📋 Next Steps:"
echo "   1. Run the scraper:"
echo "      python scrape_metro_data.py"
echo ""
echo "   2. Analyze the data:"
echo "      python analyze_fares.py"
echo ""
echo "   3. View results in:"
echo "      - data/metro_fares_clean.csv"
echo "      - visualizations/*.png"
echo "      - visualizations/*.html"
echo "=================================================="
