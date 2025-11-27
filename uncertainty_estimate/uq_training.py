import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import define_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed
set_seed(42)

dataset_size = 1000

#%% Load data and pretrained model
 
path = "dataset/"
model_path = path + "model_name.pth"
input_path = path + "train_input.npy"
std_path = path + "std_maps.npy"

# Load the numpy arrays and prepare for torch
train_std = np.load(std_path)[:dataset_size] 
train_std = torch.tensor(train_std, dtype=torch.float32).to(device)
train_std = train_std.view(train_std.size(0), -1)
    

train_input = np.load(input_path)
train_input = train_input[:dataset_size] 
train_input = torch.tensor(train_input, dtype=torch.float32).to(device)

#%% Define Model

model = define_model.BasicNetwork().to(device)

# Load pretrained weights
pre_ckpt = torch.load(model_path, map_location=device)
pre_state = pre_ckpt.state_dict() if isinstance(pre_ckpt, nn.Module) else pre_ckpt

# Load everything except fc5
backbone_state = {k: v for k, v in pre_state.items() if not k.startswith("fc5.")}
model.load_state_dict(backbone_state, strict=False)

# ------------------------------
# Define two-head structure
# ------------------------------
class TwoHeadSigmoid(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mean_head = nn.Linear(in_features, 2400)
        self.std_head  = nn.Linear(in_features, 2400)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean_part = self.sigmoid(self.mean_head(x))
        s = torch.sigmoid(self.std_head(x))          
        s = torch.clamp(s, min=1e-3)
        std_part = torch.log(s)
        return torch.cat([mean_part, std_part], dim=1)

# Replace old head
in_features = model.fc5[0].in_features  
model.fc5 = TwoHeadSigmoid(in_features).to(device)

# ------------------------------
# Initialize the heads
# ------------------------------
with torch.no_grad():
    # Copy pretrained mean weights from old fc5
    if "fc5.0.weight" in pre_state and "fc5.0.bias" in pre_state:
        model.fc5.mean_head.weight.copy_(pre_state["fc5.0.weight"])
        model.fc5.mean_head.bias.copy_(pre_state["fc5.0.bias"])
        print("✅ Loaded pretrained weights into mean_head.")
    else:
        print("⚠️ No fc5.0.* keys in checkpoint — mean_head left random.")

    # Initialize std_head freshly
    nn.init.kaiming_uniform_(model.fc5.std_head.weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.fc5.std_head.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(model.fc5.std_head.bias, -bound, bound)
    print("✅ Initialized std_head randomly (trainable).")

# ------------------------------
# Freeze everything except std_head
# ------------------------------
for p in model.parameters():
    p.requires_grad = False  # freeze all

for p in model.fc5.std_head.parameters():
    p.requires_grad = True   # train only std_head
    
for m in model.modules():
    if isinstance(m, nn.BatchNorm1d):
        m.eval()                   # freezes running stats
        m.track_running_stats = False  # optional extra safety


#%% Train the model

# --------------------------
# Batch training
# --------------------------
criterion = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.fc5.parameters(), lr=lr)

num_epochs = 30        
batch_size = 64 

N = train_input.shape[0]
num_batches = math.ceil(N / batch_size)

epoch_losses = []
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for i in range(0, N, batch_size):
        xb = train_input[i:i+batch_size]   # (B, 40)
        y_std  = train_std[i:i+batch_size]   # (B, 2400)
        y_std_safe = torch.clamp(y_std, min=1e-3)
        
        optimizer.zero_grad()

        preds = model(xb)                    # (B, 4800)
        pred_std  = preds[:, 2400:]          # last  2400 -> std prediction
        pred_std = torch.exp(pred_std)

        std_loss = criterion(pred_std, y_std_safe)

        std_loss.backward()
        optimizer.step()

        running_loss += std_loss.item()
    
    
    epoch_loss = running_loss / num_batches
    epoch_losses.append(epoch_loss)
    
    print(f"Epoch {epoch:03d} | loss: {epoch_loss:.6f}")

#%%
# --------------------------
# Save the fine-tuned head model
# --------------------------
info = "datasize_" + str(dataset_size) 
model_name = "ue_model_" +  info + ".pth"
save_path = "training_output/models/" + model_name
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# torch.save(model.state_dict(), save_path)
torch.save(model, save_path)
print(f"Saved fine-tuned model to: {save_path}")


# --------------------------
# Save the training loss plot
# --------------------------
results_dir = "training_output/results"
plot_name = "train_loss_" + info + ".png"
os.makedirs(results_dir, exist_ok=True)


plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, label="Training Loss")
plt.yscale("log")  # log-scale y-axis
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE, log scale)")
plt.title("Training Loss (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plot_path = os.path.join(results_dir, plot_name)
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()

print(f"Saved training loss plot to: {plot_path}")

#%%

