import numpy as np
import matplotlib.pyplot as plt

# Load the file
file_path = "/scratch2/mrenaudin/colorlessgreenRNNs/data/agreement/English/generated.output_epoch_40"
with open(file_path, 'r') as f:
    lines = f.readlines()

# Extract the probabilities from the file (assuming space-separated values in each line)
probs = []
for line in lines:
    values = list(map(float, line.strip().split("\t")))  # Adjust the delimiter if needed
    probs.extend(values)

# Convert to numpy array for convenience
probs = np.array(probs)

# Summary statistics
print("Min:", np.min(probs))
print("Max:", np.max(probs))
print("Mean:", np.mean(probs))
print("Standard Deviation:", np.std(probs))

# Plotting histogram of probabilities
plt.hist(probs, bins=50, range=(0, 1), alpha=0.75, color='blue')
plt.title('Distribution of Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()
