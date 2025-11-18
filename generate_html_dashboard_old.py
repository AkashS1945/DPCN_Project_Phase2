#!/usr/bin/env python3
"""
Generate comprehensive HTML dashboard for network resilience analysis
Shows all 4 removal strategies with visualizations, tables, and insights
"""

import json
import base64
from pathlib import Path
from datetime import datetime

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_html_dashboard():
    """Generate comprehensive HTML dashboard"""
    
    # Load all analysis results
    analyses = {
        'degree': {
            'name': 'Degree Centrality',
            'dir': 'degree_removal',
            'color': '#e74c3c',
            'icon': '🔗',
            'description': 'Nodes with the most connections'
        },
        'betweenness': {
            'name': 'Betweenness Centrality',
            'dir': 'betweenness_removal',
            'color': '#3498db',
            'icon': '🔀',
            'description': 'Critical routing bottleneck nodes'
        },
        'closeness': {
            'name': 'Closeness Centrality',
            'dir': 'closeness_removal',
            'color': '#2ecc71',
            'icon': '📍',
            'description': 'Most centrally located nodes'
        },
        'route': {
            'name': 'Route Removal',
            'dir': 'route_removal',
            'color': '#f39c12',
            'icon': '🛣️',
            'description': 'Most critical routes/edges'
        }
    }
    
    # Load data
    for key, analysis in analyses.items():
        result_file = Path(analysis['dir']) / 'results.json'
        with open(result_file, 'r') as f:
            analysis['data'] = json.load(f)
    
    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Resilience Analysis Dashboard - Hyderabad Urban Mobility</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: white;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        header h2 {{
            font-size: 1.3em;
            color: #7f8c8d;
            font-weight: 400;
            margin-bottom: 20px;
        }}
        
        .meta-info {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .meta-label {{
            font-size: 0.9em;
            color: #95a5a6;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .meta-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 5px;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .card-icon {{
            font-size: 2em;
            margin-right: 15px;
        }}
        
        .card-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .card-subtitle {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-top: 3px;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .metric-row:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.95em;
        }}
        
        .metric-value {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .impact-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .impact-high {{
            background: #fee;
            color: #c0392b;
        }}
        
        .impact-medium {{
            background: #fef5e7;
            color: #f39c12;
        }}
        
        .impact-low {{
            background: #eafaf1;
            color: #27ae60;
        }}
        
        .section {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #ecf0f1;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .tab {{
            padding: 12px 25px;
            border-radius: 8px;
            border: none;
            background: #ecf0f1;
            color: #7f8c8d;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .tab:hover {{
            background: #d5dbdb;
        }}
        
        .tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.5s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .rank-badge {{
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            border-radius: 50%;
            font-weight: bold;
            color: white;
            font-size: 0.9em;
        }}
        
        .rank-1 {{ background: #f39c12; }}
        .rank-2 {{ background: #95a5a6; }}
        .rank-3 {{ background: #cd7f32; }}
        .rank-other {{ background: #bdc3c7; }}
        
        .layer-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .layer-metro {{
            background: #3498db;
            color: white;
        }}
        
        .layer-mmts {{
            background: #e74c3c;
            color: white;
        }}
        
        .layer-bus {{
            background: #2ecc71;
            color: white;
        }}
        
        .layer-auto {{
            background: #f39c12;
            color: white;
        }}
        
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .viz-caption {{
            margin-top: 15px;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .insight-box {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .insight-title {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .insight-text {{
            color: #555;
            line-height: 1.8;
        }}
        
        .key-finding {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .key-finding h3 {{
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        footer {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .progress-bar {{
            background: #ecf0f1;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ max-width: 100%; }}
            .card, .section {{ box-shadow: none; border: 1px solid #ddd; }}
        }}
        
        @media (max-width: 768px) {{
            header h1 {{ font-size: 1.8em; }}
            .meta-info {{ gap: 20px; }}
            .summary-cards {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚇 Network Resilience Analysis Dashboard</h1>
            <h2>Hyderabad Urban Mobility & Public Transport Network</h2>
            <div class="meta-info">
                <div class="meta-item">
                    <div class="meta-label">Total Nodes</div>
                    <div class="meta-value">675</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Total Edges</div>
                    <div class="meta-value">3,434</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Analyses</div>
                    <div class="meta-value">4</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Generated</div>
                    <div class="meta-value" style="font-size: 1.2em;">Nov 13, 2024</div>
                </div>
            </div>
        </header>
"""
    
    # Summary cards
    html += """
        <div class="summary-cards">
"""
    
    for key, analysis in analyses.items():
        data = analysis['data']
        removal_analysis = data['removal_analysis']
        
        # Calculate impact
        baseline_eff = removal_analysis['network_efficiency'][0]
        final_eff = removal_analysis['network_efficiency'][-1]
        impact_pct = ((final_eff - baseline_eff) / baseline_eff) * 100
        
        # Determine impact level
        if abs(impact_pct) > 10:
            impact_class = 'impact-high'
            impact_text = 'HIGH IMPACT'
        elif abs(impact_pct) > 5:
            impact_class = 'impact-medium'
            impact_text = 'MEDIUM IMPACT'
        else:
            impact_class = 'impact-low'
            impact_text = 'LOW IMPACT'
        
        # Get top hub/route
        if key == 'route':
            top_item = data['top_routes'][0]
            top_name = f"{top_item['from_id']} → {top_item['to_id']}"
            top_detail = top_item.get('mode', 'unknown')
        else:
            top_item = data['top_hubs'][0]
            top_name = top_item['name']
            top_detail = top_item.get('layer', 'unknown').upper()
        
        html += f"""
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">{analysis['icon']}</div>
                    <div>
                        <div class="card-title">{analysis['name']}</div>
                        <div class="card-subtitle">{analysis['description']}</div>
                    </div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Top Critical</span>
                    <span class="metric-value">{top_name}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Layer/Mode</span>
                    <span class="metric-value">{top_detail}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Efficiency Impact</span>
                    <span class="metric-value">{impact_pct:.2f}%</span>
                </div>
                <div class="impact-badge {impact_class}">{impact_text}</div>
            </div>
"""
    
    html += """
        </div>
"""
    
    # Key Findings Section
    html += """
        <div class="section">
            <h2 class="section-title">🔍 Key Findings</h2>
            
            <div class="key-finding">
                <h3>🚨 Most Vulnerable: Betweenness Centrality Removal</h3>
                <p><strong>-11.36% efficiency loss</strong> when removing routing bottleneck nodes!</p>
                <p>Top critical nodes: <strong>HYB_MMTS (HYDERABAD)</strong>, <strong>Bus_Begumpet</strong>, <strong>SC_MMTS (SECUNDERABAD)</strong></p>
                <p>These are true <em>structural bottlenecks</em> - removing them forces longest detours across the network.</p>
            </div>
            
            <div class="insight-box">
                <div class="insight-title">💡 The Ameerpet Mystery Solved</div>
                <div class="insight-text">
                    <p><strong>Question:</strong> Why isn't Ameerpet (major Red+Blue line interchange) in all top 15 lists?</p>
                    <p><strong>Answer:</strong> It depends on the metric!</p>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>Degree Analysis:</strong> Ameerpet is <strong>#8</strong> (27 connections - high connectivity!)</li>
                        <li><strong>Betweenness Analysis:</strong> Ameerpet is ~#24 (not a routing bottleneck - good redundancy exists!)</li>
                        <li><strong>Closeness Analysis:</strong> Ameerpet is ~#22 (not most centrally located)</li>
                    </ul>
                    <p style="margin-top: 10px;"><em>Insight: Ameerpet has high connectivity but the network has good redundancy around it. Alternate routes prevent it from being a critical bottleneck!</em></p>
                </div>
            </div>
        </div>
"""
    
    # Impact Comparison
    html += """
        <div class="section">
            <h2 class="section-title">📊 Network Efficiency Impact Comparison</h2>
"""
    
    # Calculate impacts
    impacts = []
    for key, analysis in analyses.items():
        removal_analysis = analysis['data']['removal_analysis']
        baseline = removal_analysis['network_efficiency'][0]
        final = removal_analysis['network_efficiency'][-1]
        impact = abs((final - baseline) / baseline) * 100
        impacts.append((analysis['name'], impact, analysis['color']))
    
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    for name, impact, color in impacts:
        html += f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 600;">{name}</span>
                    <span style="font-weight: 600; color: {color};">{impact:.2f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {impact * 7}%; background: {color};">
                        {impact:.2f}% loss
                    </div>
                </div>
            </div>
"""
    
    html += """
        </div>
"""
    
    # Detailed Analysis Tabs
    html += """
        <div class="section">
            <h2 class="section-title">📈 Detailed Analysis by Strategy</h2>
            <div class="tabs">
"""
    
    for i, (key, analysis) in enumerate(analyses.items()):
        active = 'active' if i == 0 else ''
        html += f'<button class="tab {active}" onclick="switchTab(\'{key}\')">{analysis["icon"]} {analysis["name"]}</button>\n'
    
    html += """
            </div>
"""
    
    # Tab contents
    for i, (key, analysis) in enumerate(analyses.items()):
        active = 'active' if i == 0 else ''
        data = analysis['data']
        
        html += f"""
            <div id="{key}" class="tab-content {active}">
                <h3 style="color: {analysis['color']}; margin-bottom: 20px;">{analysis['icon']} {analysis['name']} Analysis</h3>
                <p style="font-size: 1.1em; color: #555; margin-bottom: 30px;">{analysis['description']}</p>
"""
        
        # Top 15 table
        if key == 'route':
            html += """
                <h4 style="margin-top: 30px; margin-bottom: 15px;">Top 15 Critical Routes</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>From</th>
                            <th>To</th>
                            <th>Mode</th>
                            <th>Score</th>
                            <th>Betweenness</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for route in data['top_routes'][:15]:
                rank = route['rank']
                rank_class = f'rank-{rank}' if rank <= 3 else 'rank-other'
                mode = route.get('mode', 'unknown')
                mode_class = f'layer-{mode}' if mode in ['metro', 'mmts', 'bus', 'auto'] else 'layer-bus'
                
                html += f"""
                        <tr>
                            <td><span class="rank-badge {rank_class}">{rank}</span></td>
                            <td>{route['from_id']}</td>
                            <td>{route['to_id']}</td>
                            <td><span class="layer-badge {mode_class}">{mode.upper()}</span></td>
                            <td>{route['score']:.4f}</td>
                            <td>{route.get('edge_betweenness', 0):.4f}</td>
                        </tr>
"""
            html += """
                    </tbody>
                </table>
"""
        else:
            # Determine centrality key
            if key == 'degree':
                cent_key = 'degree_centrality'
                cent_label = 'Degree'
            elif key == 'betweenness':
                cent_key = 'betweenness_centrality'
                cent_label = 'Betweenness'
            else:  # closeness
                cent_key = 'closeness_centrality'
                cent_label = 'Closeness'
            
            html += f"""
                <h4 style="margin-top: 30px; margin-bottom: 15px;">Top 15 Critical Nodes</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Node ID</th>
                            <th>Name</th>
                            <th>Layer</th>
                            <th>Degree</th>
                            <th>{cent_label}</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for hub in data['top_hubs'][:15]:
                rank = hub['rank']
                rank_class = f'rank-{rank}' if rank <= 3 else 'rank-other'
                layer = hub.get('layer', 'unknown')
                layer_class = f'layer-{layer}'
                
                html += f"""
                        <tr>
                            <td><span class="rank-badge {rank_class}">{rank}</span></td>
                            <td><strong>{hub['node_id']}</strong></td>
                            <td>{hub['name']}</td>
                            <td><span class="layer-badge {layer_class}">{layer.upper()}</span></td>
                            <td>{hub['degree']}</td>
                            <td>{hub[cent_key]:.4f}</td>
                        </tr>
"""
            html += """
                    </tbody>
                </table>
"""
        
        # Visualizations
        html += f"""
                <h4 style="margin-top: 40px; margin-bottom: 20px;">📊 Visualizations</h4>
"""
        
        # Embed combined dashboard
        dashboard_path = Path(analysis['dir']) / 'combined_dashboard.png'
        if dashboard_path.exists():
            img_data = image_to_base64(dashboard_path)
            html += f"""
                <div class="visualization">
                    <img src="data:image/png;base64,{img_data}" alt="Combined Dashboard">
                    <div class="viz-caption">All 9 network metrics showing impact of sequential removal</div>
                </div>
"""
        
        # Network efficiency plot
        eff_path = Path(analysis['dir']) / 'network_efficiency.png'
        if eff_path.exists():
            img_data = image_to_base64(eff_path)
            html += f"""
                <div class="visualization">
                    <img src="data:image/png;base64,{img_data}" alt="Network Efficiency">
                    <div class="viz-caption">Network efficiency degradation with each removal</div>
                </div>
"""
        
        html += """
            </div>
"""
    
    # Comparison visualizations
    html += """
        <div class="section">
            <h2 class="section-title">🔬 Cross-Analysis Comparison</h2>
"""
    
    # Embed comparison plots
    comp_all_path = Path('comparison_all_analyses.png')
    if comp_all_path.exists():
        img_data = image_to_base64(comp_all_path)
        html += f"""
            <div class="visualization">
                <img src="data:image/png;base64,{img_data}" alt="All Analyses Comparison">
                <div class="viz-caption">All 9 metrics compared across 4 removal strategies</div>
            </div>
"""
    
    comp_eff_path = Path('efficiency_comparison.png')
    if comp_eff_path.exists():
        img_data = image_to_base64(comp_eff_path)
        html += f"""
            <div class="visualization">
                <img src="data:image/png;base64,{img_data}" alt="Efficiency Comparison">
                <div class="viz-caption">Network efficiency degradation comparison - shows which removal strategy has biggest impact</div>
            </div>
"""
    
    html += """
        </div>
"""
    
    # Methodology section
    html += """
        <div class="section">
            <h2 class="section-title">🔬 Methodology</h2>
            
            <div class="insight-box">
                <div class="insight-title">Network Type</div>
                <div class="insight-text">
                    <strong>Undirected, Unweighted Multilayer Network</strong>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>675 nodes</strong> across 4 layers: Metro (57), MMTS (46), Bus (200), Auto (372)</li>
                        <li><strong>3,434 edges</strong> (undirected, from 6,271 total directed edges)</li>
                        <li><strong>Unweighted</strong> to ensure balanced multilayer analysis (weighted by time biases toward Metro)</li>
                    </ul>
                </div>
            </div>
            
            <div class="insight-box">
                <div class="insight-title">Centrality Measures</div>
                <div class="insight-text">
                    <ul style="margin-left: 20px;">
                        <li><strong>Degree Centrality:</strong> Number of direct connections (high connectivity hubs)</li>
                        <li><strong>Betweenness Centrality:</strong> Fraction of shortest paths passing through node (routing bottlenecks)</li>
                        <li><strong>Closeness Centrality:</strong> Average distance to all other nodes (geographic centrality)</li>
                        <li><strong>Edge Betweenness:</strong> Fraction of shortest paths using edge (critical routes)</li>
                    </ul>
                </div>
            </div>
            
            <div class="insight-box">
                <div class="insight-title">Sequential Removal Process</div>
                <div class="insight-text">
                    <p>For each analysis:</p>
                    <ol style="margin-left: 20px; margin-top: 10px;">
                        <li>Identify top 15 nodes/routes by the specific centrality metric</li>
                        <li>Remove them sequentially (1st, 2nd, ..., 15th)</li>
                        <li>After each removal, calculate 9 network metrics</li>
                        <li>Track how network degrades with each removal</li>
                        <li>Generate visualizations showing metric changes</li>
                    </ol>
                </div>
            </div>
            
            <div class="insight-box">
                <div class="insight-title">Metrics Tracked (9 total)</div>
                <div class="insight-text">
                    <ol style="margin-left: 20px;">
                        <li>Average Clustering Coefficient</li>
                        <li>Average Betweenness Centrality</li>
                        <li>Average Closeness Centrality</li>
                        <li>Average Node Degree</li>
                        <li>Average Travel Time (minutes)</li>
                        <li>Average Travel Cost (rupees)</li>
                        <li>Network Efficiency</li>
                        <li>Number of Connected Components</li>
                        <li>Largest Component Size</li>
                    </ol>
                </div>
            </div>
        </div>
"""
    
    # Recommendations
    html += """
        <div class="section">
            <h2 class="section-title">💡 Recommendations</h2>
            
            <div class="comparison-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">🛡️</div>
                        <div class="card-title">For Network Operators</div>
                    </div>
                    <ul style="margin-left: 20px; margin-top: 15px; color: #555;">
                        <li>Monitor betweenness-critical nodes (HYB_MMTS, Bus_Begumpet, SC_MMTS)</li>
                        <li>Create redundancy around high-betweenness routes</li>
                        <li>Regular capacity checks on nodes in multiple top-15 lists</li>
                    </ul>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">🏗️</div>
                        <div class="card-title">For Urban Planners</div>
                    </div>
                    <ul style="margin-left: 20px; margin-top: 15px; color: #555;">
                        <li>New connections should target low-closeness areas</li>
                        <li>Emergency planning should assume betweenness-critical nodes may fail</li>
                        <li>Metro expansion should consider bus and MMTS integration</li>
                    </ul>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">🔬</div>
                        <div class="card-title">For Future Research</div>
                    </div>
                    <ul style="margin-left: 20px; margin-top: 15px; color: #555;">
                        <li>Time-of-day analysis: How do critical nodes change during peak hours?</li>
                        <li>Cascading failures: What if failures trigger secondary failures?</li>
                        <li>Weighted betweenness: Use actual passenger flow data</li>
                    </ul>
                </div>
            </div>
        </div>
"""
    
    # Footer
    html += f"""
        <footer>
            <h3 style="margin-bottom: 15px;">📚 Analysis Summary</h3>
            <p style="color: #7f8c8d; margin-bottom: 20px;">
                This comprehensive resilience analysis examined Hyderabad's multilayer transport network
                using 4 different removal strategies to identify critical infrastructure.
            </p>
            <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px; flex-wrap: wrap;">
                <div>
                    <strong>Network:</strong> Hyderabad Urban Mobility
                </div>
                <div>
                    <strong>Date:</strong> November 13, 2024
                </div>
                <div>
                    <strong>Status:</strong> ✅ Complete
                </div>
            </div>
            <p style="margin-top: 20px; color: #95a5a6; font-size: 0.9em;">
                Generated using NetworkX, Matplotlib, and comprehensive multilayer network analysis
            </p>
        </footer>
    </div>
    
    <script>
        function switchTab(tabId) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Deactivate all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Activate selected tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    
    return html

# Generate and save
print("🎨 Generating comprehensive HTML dashboard...")
html_content = generate_html_dashboard()

output_file = 'network_resilience_dashboard.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Dashboard generated: {output_file}")
print(f"📊 File size: {len(html_content) / 1024 / 1024:.2f} MB")
print("\n" + "="*80)
print("To view the dashboard:")
print(f"  • Open in browser: xdg-open {output_file}")
print(f"  • Or navigate to: file://{Path.cwd()}/{output_file}")
print("="*80)
