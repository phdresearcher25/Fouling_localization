import os
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# ----------------------------
# Config
# ----------------------------
parent_folder = "dataset"

dataset_size =  1000  # change this for each model saved
info = "datasize_" + str(dataset_size)

model_name = "ue_model_" + info + ".pth"
model_path = "training_output/models/" + model_name

result_path = "test_results_" + info
results_dir = os.path.join("training_output", result_path)
os.makedirs(results_dir, exist_ok=True)

batch_size = 128

#%%
# ----------------------------
# Load data
# ----------------------------
test_input_path = os.path.join(parent_folder, "test_input.npy")

test_input = np.load(test_input_path)
N = test_input.shape[0]
test_input = torch.tensor(test_input, dtype=torch.float32, device=device)

#%%
# ----------------------------
# Load model
# ----------------------------
class TwoHeadSigmoid(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mean_head = nn.Linear(in_features, 2400)
        self.std_head  = nn.Linear(in_features, 2400)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean_part = self.sigmoid(self.mean_head(x))
        s = torch.sigmoid(self.std_head(x))          # (0,1)
        s = torch.clamp(s, min=1e-3)
        std_part = torch.log(s)
        return torch.cat([mean_part, std_part], dim=1)

# Load the saved fine-tuned model (saved via torch.save(model, path))
model = torch.load(model_path, map_location=device)
model.eval()

# ----------------------------
# Inference (batched)
# ----------------------------
all_pred_mean = []
all_pred_std = []

with torch.no_grad():
    for i in range(0, N, batch_size):
        xb = test_input[i:i+batch_size]
        preds = model(xb)                 # (B, 4800)
        pred_mean = preds[:, :2400]       # (B, 2400)
        pred_std  = preds[:, 2400:]       # (B, 2400)
        pred_std = torch.exp(pred_std)

        all_pred_mean.append(pred_mean.detach().cpu())
        all_pred_std.append(pred_std.detach().cpu())

pred_mean = torch.cat(all_pred_mean, dim=0)  # (N, 2400), CPU
pred_std  = torch.cat(all_pred_std,  dim=0)  # (N, 2400), CPU


pred_mean = pred_mean.numpy()   # (N,)
pred_std  = pred_std.numpy()    # (N,)

np.save(os.path.join(results_dir, "pred_mean.npy"), pred_mean)
np.save(os.path.join(results_dir, "pred_std.npy"),  pred_std)
print(f"✅ Saved arrays to: {results_dir}")



