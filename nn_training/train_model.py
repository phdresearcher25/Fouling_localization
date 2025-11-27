import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import time
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

#%%

def train_autoencoder(autoencoder, train_input, train_y, valid_input, valid_y, num_epochs, batch_size, criterion, optimizer, patience, min_delta):
    # Initialize variables to track the best model and validation loss 
    best_val_loss = float('inf')
    best_model_weights = None
    
    no_improvement_count = 0  # Counter to keep track of epochs with no improvement
    
    completed_epochs = 0

    # Define lists for loss values for each epoch
    train_losses = []
    val_losses = []

    execution_times = []
    for epoch in range(num_epochs):
        start_time = time.time()
                       
        autoencoder.train()  # Set the model to training mode
        
        # Define variable to hold total loss of an epoch
        epoch_loss = 0.0
        
        # Loop through mini-batches
        for i in range(0, len(train_input), batch_size):
            optimizer.zero_grad()  # Clear gradients
            
            # Get the current mini-batch
            batch_input = train_input[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]

            # Forward pass
            outputs = autoencoder(batch_input)

            # Compute the loss
            batch_loss = criterion(outputs, batch_y)

            # Backpropagation
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss # total loss of all batches
        
        # Epoch loss
        epoch_loss = epoch_loss / math.ceil(len(train_input)/batch_size)

        # Print training progress
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss.item()}')

        # Append the training loss to the list
        train_losses.append(epoch_loss.item())

        # Validation
        autoencoder.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_outputs = autoencoder(valid_input)
            val_loss = criterion(val_outputs, valid_y)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss.item()}')
        print('\n')

        # Append the validation loss to the list
        val_losses.append(val_loss.item())
        
        # Check if the current model is the best so far
        if best_val_loss - val_loss > min_delta:
            no_improvement_count = 0  # Reset the no improvement counter
        else:
            no_improvement_count += 1
            
        print("no imp count: ", no_improvement_count)
        print("epoch: ", epoch)
        print("best_val_loss: ", best_val_loss)
        print("val_loss: ", val_loss)
        print("best_val_loss - val_loss: ", best_val_loss - val_loss)
        print("min_delta: ", min_delta, "\n")
        
        # Check if the current model is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = autoencoder.state_dict()
        
        completed_epochs = completed_epochs + 1
        
        # Check if training should be early stopped
        if no_improvement_count >= patience:
            print(f'Early stopping after {patience} epochs with no improvement.')
            break
        
        
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        print(f"Execution time: {execution_time} seconds")
        print("\n")
        
        
    return train_losses, val_losses, best_model_weights, completed_epochs


def plot_model_history(train_losses, val_losses, completed_epochs, path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, completed_epochs + 1), train_losses, label='Train Loss')  
    plt.plot(range(1, completed_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    
    path = path + "results/" + "model_history.png"
    plt.savefig(path)    
    #plt.show() 
    

#%% Variables

path = "training_output/" 
        
#%%

# Load dataset (foulings and overlaps)
train_input = np.load('dataset/train_input.npy')
valid_input = np.load('dataset/valid_input.npy')

train_y = np.load('dataset/train_y.npy')
valid_y = np.load('dataset/valid_y.npy')

#%% Shuffle if needed

# def shuffle_arrays(arr1, arr2):
#     assert len(arr1) == len(arr2), "Arrays must be of the same length"
    
#     # Create an index array
#     indices = list(range(len(arr1)))

#     # Shuffle the indices
#     random.seed(42)
#     random.shuffle(indices)

#     # Use the shuffled indices to reorder the arrays
#     shuffled_arr1 = [arr1[i] for i in indices]
#     shuffled_arr2 = [arr2[i] for i in indices]
    
    
#     return np.array(shuffled_arr1), np.array(shuffled_arr2)

# train_input, train_y = shuffle_arrays(train_input, train_y)

#%%
# Convert NumPy arrays to PyTorch tensors
train_input = torch.Tensor(train_input)

train_y = torch.Tensor(train_y)
train_y = train_y.view(train_y.size(0), -1)


valid_input = torch.Tensor(valid_input)

valid_y = torch.Tensor(valid_y)
valid_y = valid_y.view(valid_y.size(0), -1)


train_input = train_input.to(device)
train_y = train_y.to(device)
valid_input = valid_input.to(device)
valid_y = valid_y.to(device)

#%% CREATE THE MODEL

# Create the autoencoder model
model = define_model.BasicNetwork()
model.to(device)

#%%
# Define loss function and optimizer
criterion = nn.MSELoss()
lr = 0.0001 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define batch size
batch_size = 500  

# Training loop
num_epochs = 10000

# Define patience for early stop
patience = 10 

# Define min_delta for early stop
min_delta = 0.00001 

#%% Train and save model
train_losses, val_losses, best_model_weights, completed_epochs = train_autoencoder(model, train_input, train_y, valid_input, valid_y, num_epochs, batch_size, criterion, optimizer, patience, min_delta)

torch.save(best_model_weights, path + "models/" + "model_name.pth")

#%% 
# Plot the training and validation losses
plot_model_history(train_losses, val_losses, completed_epochs, path)

#%% 


