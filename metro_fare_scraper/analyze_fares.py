"""
Metro Fare Data Analysis & Visualization
=========================================

Analyzes scraped metro fare and distance data to generate insights
and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os


class MetroFareAnalyzer:
    """Analyzer for metro fare and distance data"""
    
    def __init__(self, data_file='data/metro_fares_raw.csv'):
        self.data_file = data_file
        self.df = None
        self.stats = {}
        
        # Create output directory
        os.makedirs('visualizations', exist_ok=True)
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def load_data(self):
        """Load and clean the scraped data"""
        print("📂 Loading data...")
        
        if not os.path.exists(self.data_file):
            print(f"❌ Data file not found: {self.data_file}")
            print("   Run scrape_metro_data.py first to collect data")
            return False
        
        self.df = pd.read_csv(self.data_file)
        print(f"✅ Loaded {len(self.df)} records")
        
        # Clean data
        self.clean_data()
        
        return True
    
    def clean_data(self):
        """Clean and standardize the scraped data"""
        print("\n🧹 Cleaning data...")
        
        original_count = len(self.df)
        
        # Extract numeric fare from strings like "₹20" or "Rs. 20"
        if 'fare' in self.df.columns:
            self.df['fare_rs'] = self.df['fare'].str.extract(r'(\d+)').astype(float)
        
        # Extract numeric distance from strings like "5.2 km" or "5.2km"
        if 'distance' in self.df.columns:
            self.df['distance_km'] = self.df['distance'].str.extract(r'([\d.]+)').astype(float)
        
        # Extract travel time if available
        if 'travel_time' in self.df.columns:
            self.df['travel_time_min'] = self.df['travel_time'].str.extract(r'(\d+)').astype(float)
        
        # Remove rows with missing data
        self.df = self.df.dropna(subset=['fare_rs', 'distance_km'])
        
        # Calculate cost per km
        self.df['cost_per_km'] = self.df['fare_rs'] / self.df['distance_km']
        
        print(f"   Original records: {original_count}")
        print(f"   After cleaning: {len(self.df)}")
        print(f"   Removed: {original_count - len(self.df)}")
        
        # Save cleaned data
        clean_file = 'data/metro_fares_clean.csv'
        self.df.to_csv(clean_file, index=False)
        print(f"✅ Cleaned data saved to: {clean_file}")
    
    def calculate_statistics(self):
        """Calculate summary statistics"""
        print("\n📊 Calculating statistics...")
        
        self.stats = {
            'total_pairs': len(self.df),
            'unique_stations': len(set(self.df['from_station']) | set(self.df['to_station'])),
            'fare': {
                'min': self.df['fare_rs'].min(),
                'max': self.df['fare_rs'].max(),
                'mean': self.df['fare_rs'].mean(),
                'median': self.df['fare_rs'].median(),
                'std': self.df['fare_rs'].std()
            },
            'distance': {
                'min': self.df['distance_km'].min(),
                'max': self.df['distance_km'].max(),
                'mean': self.df['distance_km'].mean(),
                'median': self.df['distance_km'].median(),
                'std': self.df['distance_km'].std()
            },
            'cost_per_km': {
                'min': self.df['cost_per_km'].min(),
                'max': self.df['cost_per_km'].max(),
                'mean': self.df['cost_per_km'].mean(),
                'median': self.df['cost_per_km'].median()
            }
        }
        
        # Save statistics
        with open('data/fare_statistics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("FARE STATISTICS")
        print("="*70)
        print(f"Total station pairs: {self.stats['total_pairs']}")
        print(f"Unique stations: {self.stats['unique_stations']}")
        print(f"\nFare Range: ₹{self.stats['fare']['min']:.0f} - ₹{self.stats['fare']['max']:.0f}")
        print(f"Average Fare: ₹{self.stats['fare']['mean']:.2f}")
        print(f"Median Fare: ₹{self.stats['fare']['median']:.0f}")
        print(f"\nDistance Range: {self.stats['distance']['min']:.2f} - {self.stats['distance']['max']:.2f} km")
        print(f"Average Distance: {self.stats['distance']['mean']:.2f} km")
        print(f"\nCost per km: ₹{self.stats['cost_per_km']['mean']:.2f}/km")
        print("="*70)
    
    def plot_fare_distribution(self):
        """Plot fare distribution histogram"""
        print("\n📊 Creating fare distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.df['fare_rs'], bins=20, color='#764ba2', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Fare (₹)', fontsize=12)
        axes[0].set_ylabel('Number of Trips', fontsize=12)
        axes[0].set_title('Metro Fare Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['fare_rs'].mean(), color='red', linestyle='--', 
                       label=f'Mean: ₹{self.df["fare_rs"].mean():.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.df['fare_rs'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#667eea', alpha=0.7))
        axes[1].set_ylabel('Fare (₹)', fontsize=12)
        axes[1].set_title('Fare Distribution (Box Plot)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('visualizations/fare_distribution.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: visualizations/fare_distribution.png")
        plt.close()
    
    def plot_distance_distribution(self):
        """Plot distance distribution"""
        print("📊 Creating distance distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.df['distance_km'], bins=25, color='#fa709a', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Distance (km)', fontsize=12)
        axes[0].set_ylabel('Number of Trips', fontsize=12)
        axes[0].set_title('Inter-Station Distance Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['distance_km'].mean(), color='red', linestyle='--',
                       label=f'Mean: {self.df["distance_km"].mean():.2f} km')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_dist = np.sort(self.df['distance_km'])
        cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist) * 100
        axes[1].plot(sorted_dist, cumulative, color='#764ba2', linewidth=2)
        axes[1].set_xlabel('Distance (km)', fontsize=12)
        axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
        axes[1].set_title('Cumulative Distance Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/distance_distribution.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: visualizations/distance_distribution.png")
        plt.close()
    
    def plot_fare_vs_distance(self):
        """Scatter plot: Fare vs Distance with regression"""
        print("📊 Creating fare vs distance correlation plot...")
        
        fig = plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.scatter(self.df['distance_km'], self.df['fare_rs'], 
                   alpha=0.5, s=30, color='#667eea', edgecolors='black', linewidths=0.5)
        
        # Add regression line
        z = np.polyfit(self.df['distance_km'], self.df['fare_rs'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['distance_km'], p(self.df['distance_km']), 
                "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        plt.xlabel('Distance (km)', fontsize=14)
        plt.ylabel('Fare (₹)', fontsize=14)
        plt.title('Metro Fare vs Distance Correlation', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Calculate correlation
        correlation = self.df['distance_km'].corr(self.df['fare_rs'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('visualizations/fare_vs_distance.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: visualizations/fare_vs_distance.png")
        plt.close()
    
    def plot_cost_per_km(self):
        """Plot cost per kilometer analysis"""
        print("📊 Creating cost-per-km analysis...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Distribution
        axes[0].hist(self.df['cost_per_km'], bins=30, color='#4facfe', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Cost per km (₹/km)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Cost per Kilometer Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['cost_per_km'].mean(), color='red', linestyle='--',
                       label=f'Mean: ₹{self.df["cost_per_km"].mean():.2f}/km')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cost per km vs distance
        axes[1].scatter(self.df['distance_km'], self.df['cost_per_km'],
                       alpha=0.5, s=30, color='#f093fb', edgecolors='black', linewidths=0.5)
        axes[1].set_xlabel('Distance (km)', fontsize=12)
        axes[1].set_ylabel('Cost per km (₹/km)', fontsize=12)
        axes[1].set_title('Cost Efficiency vs Distance', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/cost_per_km_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: visualizations/cost_per_km_analysis.png")
        plt.close()
    
    def create_fare_heatmap(self):
        """Create interactive fare heatmap"""
        print("📊 Creating interactive fare heatmap...")
        
        # Create pivot table
        pivot = self.df.pivot_table(
            values='fare_rs',
            index='from_station',
            columns='to_station',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot,
            labels=dict(x="To Station", y="From Station", color="Fare (₹)"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale='RdYlGn_r',
            title='Metro Fare Matrix - All Station Pairs'
        )
        
        fig.update_layout(
            width=1200,
            height=1200,
            font=dict(size=10)
        )
        
        fig.write_html('visualizations/fare_heatmap_interactive.html')
        print("   ✅ Saved: visualizations/fare_heatmap_interactive.html")
    
    def create_dashboard(self):
        """Create comprehensive dashboard"""
        print("📊 Creating comprehensive dashboard...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Fare Distribution',
                'Distance Distribution',
                'Fare vs Distance',
                'Top 10 Most Expensive Routes'
            ),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Fare distribution
        fig.add_trace(
            go.Histogram(x=self.df['fare_rs'], name='Fare', marker_color='#764ba2'),
            row=1, col=1
        )
        
        # 2. Distance distribution
        fig.add_trace(
            go.Histogram(x=self.df['distance_km'], name='Distance', marker_color='#fa709a'),
            row=1, col=2
        )
        
        # 3. Fare vs Distance scatter
        fig.add_trace(
            go.Scatter(
                x=self.df['distance_km'],
                y=self.df['fare_rs'],
                mode='markers',
                marker=dict(color='#667eea', size=5, opacity=0.6),
                name='Routes'
            ),
            row=2, col=1
        )
        
        # 4. Top expensive routes
        top_routes = self.df.nlargest(10, 'fare_rs')
        route_labels = [f"{row['from_station'][:10]}→{row['to_station'][:10]}" 
                       for _, row in top_routes.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=top_routes['fare_rs'],
                y=route_labels,
                orientation='h',
                marker_color='#4facfe',
                name='Fare'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Hyderabad Metro Fare Analysis Dashboard",
            showlegend=False,
            height=800,
            width=1400
        )
        
        fig.write_html('visualizations/analysis_dashboard.html')
        print("   ✅ Saved: visualizations/analysis_dashboard.html")
    
    def generate_report(self):
        """Generate text report"""
        print("\n📝 Generating analysis report...")
        
        report = f"""
{'='*70}
HYDERABAD METRO FARE & DISTANCE ANALYSIS REPORT
{'='*70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
{'-'*70}
Total Station Pairs Analyzed: {self.stats['total_pairs']:,}
Unique Stations: {self.stats['unique_stations']}

FARE ANALYSIS
{'-'*70}
Minimum Fare: ₹{self.stats['fare']['min']:.0f}
Maximum Fare: ₹{self.stats['fare']['max']:.0f}
Average Fare: ₹{self.stats['fare']['mean']:.2f}
Median Fare: ₹{self.stats['fare']['median']:.0f}
Standard Deviation: ₹{self.stats['fare']['std']:.2f}

DISTANCE ANALYSIS
{'-'*70}
Minimum Distance: {self.stats['distance']['min']:.2f} km
Maximum Distance: {self.stats['distance']['max']:.2f} km
Average Distance: {self.stats['distance']['mean']:.2f} km
Median Distance: {self.stats['distance']['median']:.2f} km
Standard Deviation: {self.stats['distance']['std']:.2f} km

COST EFFICIENCY
{'-'*70}
Average Cost per km: ₹{self.stats['cost_per_km']['mean']:.2f}/km
Minimum Cost per km: ₹{self.stats['cost_per_km']['min']:.2f}/km
Maximum Cost per km: ₹{self.stats['cost_per_km']['max']:.2f}/km

TOP 5 MOST EXPENSIVE ROUTES
{'-'*70}
"""
        
        top_5 = self.df.nlargest(5, 'fare_rs')[['from_station', 'to_station', 'fare_rs', 'distance_km']]
        for i, row in top_5.iterrows():
            report += f"{row['from_station']} → {row['to_station']}: ₹{row['fare_rs']:.0f} ({row['distance_km']:.2f} km)\n"
        
        report += f"\nTOP 5 LONGEST ROUTES\n{'-'*70}\n"
        top_dist = self.df.nlargest(5, 'distance_km')[['from_station', 'to_station', 'fare_rs', 'distance_km']]
        for i, row in top_dist.iterrows():
            report += f"{row['from_station']} → {row['to_station']}: {row['distance_km']:.2f} km (₹{row['fare_rs']:.0f})\n"
        
        report += f"\n{'='*70}\n"
        
        # Save report
        with open('visualizations/analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print("✅ Report saved to: visualizations/analysis_report.txt")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*70)
        print("  METRO FARE ANALYSIS")
        print("="*70)
        
        # Load data
        if not self.load_data():
            return
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Generate all visualizations
        self.plot_fare_distribution()
        self.plot_distance_distribution()
        self.plot_fare_vs_distance()
        self.plot_cost_per_km()
        self.create_fare_heatmap()
        self.create_dashboard()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("Output files:")
        print("  📊 visualizations/fare_distribution.png")
        print("  📊 visualizations/distance_distribution.png")
        print("  📊 visualizations/fare_vs_distance.png")
        print("  📊 visualizations/cost_per_km_analysis.png")
        print("  🌐 visualizations/fare_heatmap_interactive.html")
        print("  🌐 visualizations/analysis_dashboard.html")
        print("  📝 visualizations/analysis_report.txt")
        print("  💾 data/metro_fares_clean.csv")
        print("  💾 data/fare_statistics.json")
        print("="*70)


def main():
    """Main execution"""
    analyzer = MetroFareAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
