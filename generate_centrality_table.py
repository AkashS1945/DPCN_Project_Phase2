import json

# Load centrality data
with open('centrality_analysis_complete.json', 'r') as f:
    data = json.load(f)

# Create summary tables
output = []

output.append('=' * 80)
output.append('COMPREHENSIVE CENTRALITY ANALYSIS - TOP 10 NODES SUMMARY')
output.append('Hyderabad Urban Mobility & Public Transport Networks')
output.append('=' * 80)
output.append('')

# 1. Degree Centrality
output.append('1. DEGREE CENTRALITY')
output.append('-' * 80)
output.append(f"{'Rank':<6} {'Node Name':<40} {'Degree':<10} {'Normalized':<12}")
output.append('-' * 80)
for i, node in enumerate(data['degree_centrality']['top_10'], 1):
    output.append(f"{i:<6} {node['name']:<40} {node['degree']:<10} {node['degree_centrality']:.6f}")
output.append('')
output.append('')

# 2. Betweenness Centrality
output.append('2. BETWEENNESS CENTRALITY')
output.append('-' * 80)
output.append(f"{'Rank':<6} {'Node Name':<40} {'Betweenness':<15}")
output.append('-' * 80)
for i, node in enumerate(data['betweenness_centrality']['top_10'], 1):
    output.append(f"{i:<6} {node['name']:<40} {node['betweenness']:.8f}")
output.append('')
output.append('')

# 3. Closeness Centrality
output.append('3. CLOSENESS CENTRALITY')
output.append('-' * 80)
output.append(f"{'Rank':<6} {'Node Name':<40} {'Closeness':<15}")
output.append('-' * 80)
for i, node in enumerate(data['closeness_centrality']['top_10'], 1):
    output.append(f"{i:<6} {node['name']:<40} {node['closeness']:.8f}")
output.append('')
output.append('')

# 4. Eigenvector Centrality
output.append('4. EIGENVECTOR CENTRALITY')
output.append('-' * 80)
output.append(f"{'Rank':<6} {'Node Name':<40} {'Eigenvector':<15}")
output.append('-' * 80)
for i, node in enumerate(data['eigenvector_centrality']['top_10'], 1):
    output.append(f"{i:<6} {node['name']:<40} {node['eigenvector']:.8f}")
output.append('')
output.append('')

# 5. Clustering Coefficient
output.append('5. CLUSTERING COEFFICIENT')
output.append('-' * 80)
output.append(f"{'Rank':<6} {'Node Name':<40} {'Clustering':<12} {'Degree':<8}")
output.append('-' * 80)
for i, node in enumerate(data['clustering_coefficient']['top_10'], 1):
    output.append(f"{i:<6} {node['name']:<40} {node['clustering']:.6f}   {node['degree']:<8}")
output.append('')
output.append(f"Network Average Clustering Coefficient: {data['clustering_coefficient']['average']:.6f}")
output.append('')
output.append('=' * 80)

# Write to file
with open('CENTRALITY_SUMMARY_TABLE.txt', 'w') as f:
    f.write('\n'.join(output))

print('\n'.join(output))
print('\n✅ Summary saved to: CENTRALITY_SUMMARY_TABLE.txt')
