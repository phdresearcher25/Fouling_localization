import numpy as np
import utility

#%%

def calculate_overlaps(B, data):
    overlaps = (B @ data.reshape(len(data),-1).T).T
    
    return overlaps

# get fouling/clean ratio
def get_ratio(overlaps,C):

    overlap_ratio = np.exp(-overlaps/C) 
    
    return overlap_ratio

#%%

Gx = utility.Gx
Gy = utility.Gy

# trajectory maps B
B = np.load('B_Dx_Dy/B.npy')

# Get distances matrix D
Dx = np.load('B_Dx_Dy/Dx.npy')
Dy = np.load('B_Dx_Dy/Dy.npy')

train_y = np.load('dataset/train_y.npy')
valid_y = np.load('dataset/valid_y.npy')
test_y = np.load('dataset/test_y.npy')



train_input = calculate_overlaps(B, train_y)
valid_input = calculate_overlaps(B, valid_y)
test_input = calculate_overlaps(B, test_y)

train_input = get_ratio(train_input, 10)
valid_input = get_ratio(valid_input, 10)
test_input = get_ratio(test_input, 10)


np.save('dataset/train_input.npy', train_input)
np.save('dataset/valid_input.npy', valid_input)
np.save('dataset/test_input.npy', test_input)


#%%
