#!/usr/bin/env python3
"""
Update Metro HTML with Critical Stations Timing Analysis Section
"""

import json
from pathlib import Path

def generate_timing_section():
    """Generate HTML for critical stations timing analysis"""
    
    # Load the enhanced report
    report_file = Path("../analysis_output/enhanced_metro_report.json")
    with open(report_file, 'r') as f:
        data = json.load(f)
    
    # Build timing table HTML
    timing_rows = ""
    for hub in data['top_critical_hubs']:
        timing_rows += f"""
                    <tr>
                        <td><strong>{hub['name']}</strong></td>
                        <td>{hub['total_services']}</td>
                        <td>{hub['first_service']}</td>
                        <td>{hub['last_service']}</td>
                        <td>{hub['avg_frequency_minutes']} min</td>
                        <td><span style="font-size: 0.9em">{hub['peak_hours']}</span></td>
                    </tr>"""
    
    # Build busiest stations cards
    busiest_cards = ""
    for i, station in enumerate(data['timing_summary']['busiest_stations'], 1):
        busiest_cards += f"""
                <div class="metric-card">
                    <h3>#{i}</h3>
                    <p><strong>{station['name']}</strong></p>
                    <p style="font-size: 0.9em; margin-top: 10px;">{station['total_services']} services/day</p>
                    <p style="font-size: 0.8em;">{station['first_service']} - {station['last_service']}</p>
                    <p style="font-size: 0.8em;">Every {station['avg_frequency_minutes']} min</p>
                </div>"""
    
    html_section = f"""
        
        <!-- Critical Stations - Timing Analysis -->
        <div class="section">
            <h2>⏰ Critical Stations - Importance & Timing Analysis</h2>
            <p>Analysis of the most important stations based on network centrality, service frequency, and operational hours.</p>
            
            <table class="resilience-table">
                <thead>
                    <tr>
                        <th>Station</th>
                        <th>Daily Services</th>
                        <th>First Service</th>
                        <th>Last Service</th>
                        <th>Avg Frequency</th>
                        <th>Peak Hours</th>
                    </tr>
                </thead>
                <tbody>{timing_rows}
                </tbody>
            </table>
            
            <div class="key-findings">
                <h3>📈 Service Patterns</h3>
                <ul>
                    <li><strong>Ameerpet:</strong> 2,292 services/day, avg 0.5 min frequency - Major interchange hub</li>
                    <li><strong>Mahatma Gandhi Bus Station:</strong> 1,667 services/day, avg 0.7 min frequency - Second busiest</li>
                    <li><strong>Parade Ground:</strong> 1,629 services/day, avg 0.7 min frequency - Major hub</li>
                    <li><strong>Peak Hours:</strong> Morning (6-10 AM) and Evening (5-9 PM) see 219-615 services</li>
                </ul>
            </div>
        </div>
"""
    
    return html_section

def insert_timing_section_into_html():
    """Insert the timing section into the existing Metro HTML"""
    
    html_file = Path("../analysis_output/index.html")
    
    # Read existing HTML
    with open(html_file, 'r') as f:
        html_content = f.read()
    
    # Remove any existing Critical Stations sections
    import re
    # Pattern to match the entire Critical Stations section
    pattern = r'<!-- Critical Stations - Timing Analysis -->.*?</div>\s*</div>\s*(?=\s*<!--|\s*<div class="section">)'
    html_content = re.sub(pattern, '', html_content, flags=re.DOTALL)
    
    # Generate new section
    timing_section = generate_timing_section()
    
    # Find where to insert (before Resilience Analysis section)
    insert_marker = '<!-- Resilience Analysis Section -->'
    
    if insert_marker in html_content:
        # Insert before resilience section
        html_content = html_content.replace(insert_marker, timing_section + '\n        ' + insert_marker)
    else:
        # Fallback: insert before recommendations
        insert_marker = '<!-- Recommendations Section -->'
        if insert_marker in html_content:
            html_content = html_content.replace(insert_marker, timing_section + '\n        ' + insert_marker)
        else:
            print("⚠️  Could not find insertion point")
            return False
    
    # Save updated HTML
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Updated {html_file} with Critical Stations timing analysis")
    return True

if __name__ == "__main__":
    print("="*70)
    print("UPDATING METRO HTML WITH TIMING ANALYSIS")
    print("="*70)
    
    success = insert_timing_section_into_html()
    
    if success:
        print("\n✓ Metro HTML successfully updated!")
        print("✓ New section: 'Critical Stations - Importance & Timing Analysis'")
        print("✓ Open ../analysis_output/index.html to view")
    else:
        print("\n⚠️  Update failed - check file paths")
    
    print("="*70)
