#!/usr/bin/env python3
import itertools

# Define your parameter arrays
HIDDEN_DIMS = [1024]
NUM_HEADS = [1]
TEMPERATURES = [0.1, 0.01, 0.001]
GUMBEL_SOFTMAX_OPTIONS = [True]

# Output file
OUTPUT_FILE = "grid_search/grid_params_simple.txt"

# Generate all combinations using itertools.product
combinations = list(itertools.product(
    HIDDEN_DIMS,
    NUM_HEADS,
    TEMPERATURES,
    GUMBEL_SOFTMAX_OPTIONS
))

# Write to file
with open(OUTPUT_FILE, 'w') as f:
    for hidden_dim, num_heads, temp, gumbel_softmax in combinations:
        # Convert boolean to lowercase string for consistency
        gumbel_str = str(gumbel_softmax).lower()
        f.write(f"{hidden_dim},{num_heads},{temp},{gumbel_str}\n")

print(f"Generated {len(combinations)} parameter combinations in {OUTPUT_FILE}")
print("First few lines:")
with open(OUTPUT_FILE, 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(line.strip())
        else:
            break