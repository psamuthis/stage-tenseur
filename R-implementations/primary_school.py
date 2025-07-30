import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker_hals
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set backend
tl.set_backend('numpy')

# --- Load and preprocess data ---
dataset = pd.read_csv(
    "/home/rousseau/Documents/datasets/primary-school/primaryschool.csv",
    header=None, sep="\t"
)
dataset.columns = ["time", "p1", "p2", "p1_class", "p2_class"]

# --- Map unique node IDs to indices ---
all_nodes = sorted(set(dataset["p1"]).union(dataset["p2"]))
id_to_index = {str(node): idx for idx, node in enumerate(all_nodes)}

dataset["p1_str"] = dataset["p1"].astype(str)
dataset["p2_str"] = dataset["p2"].astype(str)
dataset["p1_mapped"] = dataset["p1_str"].map(id_to_index)
dataset["p2_mapped"] = dataset["p2_str"].map(id_to_index)

# --- Bin time ---
bin_size = 300  # 5 minutes
dataset["time_bin"] = dataset["time"] // bin_size
time_bins = sorted(dataset["time_bin"].unique())
matrix_dim = len(all_nodes)
time_samples_count = len(time_bins)

# --- Build tensor (person × person × time) ---
tensor_array = np.zeros((matrix_dim, matrix_dim, time_samples_count))

for i, current_bin in enumerate(time_bins):
    edges = dataset[dataset["time_bin"] == current_bin]
    for _, row in edges.iterrows():
        a, b = row["p1_mapped"], row["p2_mapped"]
        tensor_array[a, b, i] = 1
        tensor_array[b, a, i] = 1  # symmetric

# --- TensorLy tensor ---
tensor = tl.tensor(tensor_array)

# --- Perform HALS-based non-negative Tucker decomposition ---
core, factors = non_negative_tucker_hals(
    tensor,
    rank=[13, 13, 4],
    n_iter_max=100,
    tol=1e-5,
    verbose=True
)

# --- Reorder factor matrix columns by class ---
# Combine ID/class into a single mapping
person_classes = pd.concat([
    dataset[["p1_str", "p1_class"]].rename(columns={"p1_str": "id", "p1_class": "cls"}),
    dataset[["p2_str", "p2_class"]].rename(columns={"p2_str": "id", "p2_class": "cls"})
]).drop_duplicates().set_index("id")["cls"].to_dict()

# Group IDs by class
class_groups = defaultdict(list)
for pid, cls in person_classes.items():
    class_groups[cls].append(pid)

# Define class order (from paper)
class_order = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B", "Teachers"]

# Build ordered list of node IDs
ordered_ids = []
for cls in class_order:
    ordered_ids.extend(sorted(class_groups[cls]))

# Map to tensor indices
ordered_indices = [id_to_index[pid] for pid in ordered_ids if pid in id_to_index]

# Get mode-1 factor matrix and reorder its rows
mode1_matrix = factors[0]  # shape: (242, 13)
reordered_matrix = mode1_matrix[ordered_indices, :]  # shape: (242, 13)
heatmap_data = reordered_matrix.T  # shape: (13, 242)

# Normalize rows of heatmap data between 0 and 1 for better contrast
heatmap_data_norm = (heatmap_data - heatmap_data.min(axis=1, keepdims=True)) / \
                    (heatmap_data.max(axis=1, keepdims=True) - heatmap_data.min(axis=1, keepdims=True) + 1e-10)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_norm, cmap="Spectral", xticklabels=False, yticklabels=True)
plt.xlabel("Individuals (grouped by class)")
plt.ylabel("Rank Components")
plt.title("Mode-1 Factor Matrix (Non-negative Tucker - HALS)\nOrdered by Class, Normalized")
plt.tight_layout()
plt.savefig("/tmp/factor_matrix_heatmap_normalized.png")
plt.close()