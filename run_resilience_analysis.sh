#!/bin/bash
# Quick-start script for running resilience analysis

echo "=============================================="
echo "Network Resilience Analysis - Quick Start"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install pandas networkx matplotlib seaborn numpy scikit-learn
else
    echo "Virtual environment found. Activating..."
    source venv/bin/activate
fi

# Check if required data files exist
if [ ! -f "nodes.csv" ] || [ ! -f "edges.csv" ]; then
    echo ""
    echo "ERROR: Required data files not found!"
    echo "Please ensure nodes.csv and edges.csv exist in this directory."
    exit 1
fi

echo ""
echo "Running resilience analysis..."
echo "This may take 2-3 minutes..."
echo ""

python resilience_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Analysis Complete!"
    echo "=============================================="
    echo ""
    echo "Generated files:"
    echo "  📊 resilience_analysis_results.json"
    echo "  📁 resilience_plots/ (10 visualization files)"
    echo ""
    echo "View the plots:"
    echo "  cd resilience_plots"
    echo "  ls -lh"
    echo ""
    echo "Read the documentation:"
    echo "  cat RESILIENCE_ANALYSIS_README.md"
    echo ""
else
    echo ""
    echo "ERROR: Analysis failed. Check the error messages above."
    exit 1
fi
