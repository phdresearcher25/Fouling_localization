import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import utility
import define_model


#%%
    
def save_map(index, test_y, test_outputs, path, data_type):
    
    test_y = test_y.view(len(test_y), utility.Gy, utility.Gx)
    test_y = test_y.unsqueeze(1)

    test_outputs = test_outputs.view(len(test_outputs), utility.Gy, utility.Gx)
    test_outputs = test_outputs.unsqueeze(1)
    
    plt.figure(figsize=(8,4))
        
    plt.subplot(121)
    plt.imshow(test_y[index].numpy()[0], vmin=0, vmax=1)
    plt.colorbar()
    plt.title('True fouling')
    
    plt.subplot(122)
    plt.imshow(test_outputs[index].numpy()[0], vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Reconstructed fouling')
    
    plt.tight_layout()
    
    path = path + "results/" + data_type + "_true_reconstructed_fouling_" + str(index) + ".png"
    plt.savefig(path, dpi=300)
    
    #plt.show()
       
    
def save_label(model, index, input_data, B, path, data_type):
    #predict
    with torch.no_grad():
        F_map = model(input_data[[index]])
        
    # Flatten F_map[0] to shape (1, 48 * 50)
    F_map_flat = F_map[0].view(1, -1)

    # Convert F_map_flat to a NumPy array
    F_map_flat_numpy = F_map_flat.numpy().T

    plt.figure(figsize=(8,4))

    plt.plot(np.exp(-B @ F_map_flat_numpy/1), label = 'predicted input')
    plt.plot(input_data[[index]][0], label = 'true input')
    plt.legend()

    path = path + "results/" + data_type + "_true_predicted_input_" + str(index) + ".png"
    plt.savefig(path, dpi=300)
    
    #plt.show()
    
    
def calculate_loss(model, criterion, test_input, test_y, path, data_type):
    # Forward pass on the entire test set
    with torch.no_grad():
        test_outputs = model(test_input)
        test_loss = criterion(test_outputs, test_y)
      
    # To save results
    folder_path = path + "results/" 
    file_name = 'results.txt'  
    full_file_path = os.path.join(folder_path, file_name)
    
    # Check if the file already exists
    if os.path.isfile(full_file_path):
        mode = 'a'  # Append mode
    else:
        mode = 'w'  # Write mode if the file doesn't exist
        
    with open(full_file_path, mode) as file:
        text_to_save = data_type + " loss: " + str(float(test_loss))
        file.write(text_to_save + '\n')
    
    return test_outputs, test_loss

#%% Variables
path = "training_output/" 

criterion = nn.MSELoss()

B = np.load('B_Dx_Dy/B.npy')

#%% Load Test data and Convert torch tensor
test_y = np.load('dataset/test_y.npy') 
test_input = np.load('dataset/test_input.npy')

test_input = torch.Tensor(test_input)
test_y = torch.Tensor(test_y)
test_y = test_y.view(test_y.size(0), -1)


# Load valid and train
valid_y = np.load('dataset/valid_y.npy')
valid_input = np.load('dataset/valid_input.npy')

valid_input = torch.Tensor(valid_input)
valid_y = torch.Tensor(valid_y)
valid_y = valid_y.view(valid_y.size(0), -1)


train_y = np.load('dataset/train_y.npy')
train_input = np.load('dataset/train_input.npy')

train_input = torch.Tensor(train_input)
train_y = torch.Tensor(train_y)
train_y = train_y.view(train_y.size(0), -1)


# Load model
model = define_model.BasicNetwork()
model.load_state_dict(torch.load('training_output/models/model_name.pth'))
model.eval()

#%%

with torch.no_grad():
    test_outputs = model(test_input)
    valid_outputs = model(valid_input)
    train_outputs = model(train_input)


def calculate_iou(array1, array2, threshold=0.5):
    # Step 1: Threshold the arrays
    binary_array1 = (array1 >= threshold).float()
    binary_array2 = (array2 >= threshold).float()

    # Step 2: Compute the Intersection and Union
    intersection = (binary_array1 + binary_array2 == 2).float().sum()
    union = (binary_array1 + binary_array2 >= 1).float().sum()

    # Step 3: Calculate IoU
    iou = intersection / union
    
    # Step 4: Calculate IoU, handle division by zero
    if union == 0:
        return 0.0
    
    return iou

def average_iou(arrays1, arrays2, threshold=0.5):
    total_iou = 0.0
    batch_size = arrays1.shape[0]

    for i in range(batch_size):
        array1 = arrays1[i]
        array2 = arrays2[i]
        iou = calculate_iou(array1, array2, threshold)
        total_iou += iou

    # Step 4: Compute the average IoU
    average_iou = total_iou / batch_size
    return average_iou.mean().item()


test_avg_iou = average_iou(test_outputs, test_y, threshold=0.4)
print(f'test Average IoU: {test_avg_iou}')

valid_avg_iou = average_iou(valid_outputs, valid_y, threshold=0.4)
print(f'valid Average IoU: {valid_avg_iou}')

train_avg_iou = average_iou(train_outputs, train_y, threshold=0.4)
print(f'train Average IoU: {train_avg_iou}')


# To save results
folder_path = path + "results/" 
file_name = 'IoU_500.txt'  
full_file_path = os.path.join(folder_path, file_name)

# Check if the file already exists
if os.path.isfile(full_file_path):
    mode = 'a'  # Append mode
else:
    mode = 'w'  # Write mode if the file doesn't exist
    
with open(full_file_path, mode) as file:
    text_to_save = " test_avg_iou: " + str(float(test_avg_iou))
    file.write(text_to_save + '\n')
    
    text_to_save = " valid_avg_iou: " + str(float(valid_avg_iou))
    file.write(text_to_save + '\n')
    
    text_to_save = " train_avg_iou: " + str(float(train_avg_iou))
    file.write(text_to_save + '\n')

#%%
# Test
test_outputs, test_GP_loss = calculate_loss(model, criterion, test_input, test_y, path, "Test")

# Valid
valid_outputs, valid_loss = calculate_loss(model, criterion, valid_input, valid_y, path, "Valid")

# Train
train_outputs, train_loss = calculate_loss(model, criterion, train_input, train_y, path, "Train")


for index in range(20):
    save_map(index, test_y, test_outputs, path, "Test")  
    save_label(model, index, test_input, B, path, "Test")


#%%


