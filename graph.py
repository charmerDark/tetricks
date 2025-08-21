import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
# Replace 'your_file.csv' with your actual CSV file path
df = pd.read_csv('data.csv')

# Assuming your CSV columns are named: 'kernel name', 'naive code runtime', 'optimised code runtime'
# Adjust column names if they're different in your CSV
kernel_names = df['kernel name']
naive_runtimes = df['naive code runtime']
optimized_runtimes = df['optimised code runtime']

# Calculate speedup (naive runtime / optimized runtime)
speedup = naive_runtimes / optimized_runtimes

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Create bar positions
x = np.arange(len(kernel_names))
width = 0.35

# Create bars for naive and optimized runtimes
bars1 = ax.bar(x - width/2, naive_runtimes, width, label='Naive Code', color='tab:orange')
bars2 = ax.bar(x + width/2, optimized_runtimes, width, label='Optimized Code', color='tab:blue')

# Customize the plot

plt.xticks(fontsize=16)
ax.set_xlabel('Kernel Name', fontsize=12)
ax.set_ylabel('Runtime (cycles)', fontsize=12)
ax.set_title('Runtime Comparison: Naive vs Optimized Code', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(kernel_names, rotation=45, ha='right')
ax.legend()

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()

add_value_labels(bars1)
add_value_labels(bars2)

# Add speedup annotations with percentage
for i, (naive, optimized, speedup_val) in enumerate(zip(naive_runtimes, optimized_runtimes, speedup)):
    # Calculate percentage improvement
    percentage_improvement = ((naive - optimized) / naive) * 100
    # Position annotation on top of optimized bar (right bar)
    ax.annotate(f'{percentage_improvement:.1f}%', 
                xy=(i + width/2, optimized + optimized * 0.02),
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='green')

# Improve layout
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)

plt.show()

#plt.savefig('runtime_comparison.png', dpi=300, bbox_inches='tight')

# Print speedup summary
print("\nSpeedup Summary:")
print("-" * 50)
for kernel, naive, optimized, speedup_val in zip(kernel_names, naive_runtimes, optimized_runtimes, speedup):
    percentage_improvement = ((naive - optimized) / naive) * 100
    print(f"{kernel}: {speedup_val:.2f}x faster ({percentage_improvement:.1f}% improvement)")