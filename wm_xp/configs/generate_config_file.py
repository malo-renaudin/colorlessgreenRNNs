#!/usr/bin/env python3
import itertools

def generate_config(hidden_dim_range, heads_range, temp_range, gumbel_softmax_options, output):


    # Generate all combinations using itertools.product
    combinations = list(itertools.product(
        hidden_dim_range,
        heads_range,
        temp_range,
        gumbel_softmax_options
    ))

    # Write to file
    with open(output, 'w') as f:
        for hidden_dim, num_heads, temp, gumbel_softmax in combinations:
            # Convert boolean to lowercase string for consistency
            gumbel_str = str(gumbel_softmax).lower()
            f.write(f"{hidden_dim},{num_heads},{temp},{gumbel_str}\n")

    print(f"Generated {len(combinations)} parameter combinations in {output}")
    print("First few lines:")
    with open(output, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(line.strip())
            else:
                break