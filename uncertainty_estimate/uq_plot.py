import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Path to the directory containing your folders
base_path = 'training_output'

# List all entries that match the pattern "test_results_datasize_<number>"
folders = [
    name for name in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, name)) and
       re.match(r"test_results_datasize_\d+", name)
]

# Sort folders numerically based on the number in the name
folders_sorted = sorted(
    folders,
    key=lambda x: int(re.search(r"(\d+)", x).group())
)

# Print them in ascending order (from 100 to 1000)
for folder in folders_sorted:
    print(folder)
    
    
#%%
# 2) For each folder, load the arrays
sizes = []
means = []
stds = []

for folder in folders_sorted:
    size = int(re.search(r"(\d+)$", folder).group(1))
    folder_path = os.path.join(base_path, folder)
    mean_file = os.path.join(folder_path, 'pred_mean.npy')
    std_file  = os.path.join(folder_path, 'pred_std.npy')

    if not (os.path.exists(mean_file) and os.path.exists(std_file)):
        print(f"Skipping {folder}: missing .npy files")
        continue

    mean_arr = np.load(mean_file)  # array of pixel means for this training size
    std_arr  = np.load(std_file)   # array of per-pixel stds for this training size
    
    means.append(mean_arr)
    stds.append(std_arr)
    sizes.append(size)

# Convert to numpy arrays
sizes = np.array(sizes)

#%%    
x_arr = np.load("dataset/test_y.npy") # ground truth mean maps
x_arr = x_arr.reshape(x_arr.shape[0], -1)


joint_log_probs = []
sum_joint_log_probs = []


for i in range(len(folders_sorted)):

    # Ensure numerical stability (avoid divide-by-zero)
    eps = 1e-3
    std_safe = np.clip(stds[i], eps, None)
    
    # Compute per-pixel log probabilities
    log_prob = -0.5 * np.log(2 * np.pi) \
               - np.log(std_safe) \
               - ((x_arr - means[i]) ** 2) / (2 * std_safe ** 2)
    
    
    joint_log_prob = np.sum(log_prob, axis=1)    
    joint_log_probs.append(joint_log_prob) 


#%% Plot mean joint log-likelihood vs training size

mean_ll_per_size = [np.mean(jlp) for jlp in joint_log_probs]
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(
    sizes,
    mean_ll_per_size,
    marker='o',
    linestyle='-',
    linewidth=2,
    markersize=6,
    label='Mean'
)

ax.set_xlabel('Training size', fontsize=12)
ax.set_ylabel('Mean joint log-likelihood', fontsize=12)
# ax.set_title('Mean Joint Log-Likelihood vs Training Size', fontsize=14)
ax.grid(alpha=0.3)
ax.legend()

# Add all training sizes on x-axis
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)

# Add a small margin around the x-limits
x_min, x_max = sizes.min(), sizes.max()
x_margin = 0.05 * (x_max - x_min)
ax.set_xlim(x_min - x_margin, x_max + x_margin)

plt.grid(False)
plt.tight_layout()
plt.show()
