import numpy as np
import scipy
import utility

#%%
def sigmoid(x, scale=1):
    return 1 / (1 + np.exp(-x * scale))


def generate_fouling(Dx, Dy, Gx,Gy, N_samples):
    
    lengthscale_x = 5 
    lengthscale_y = 14
      
    K = np.exp(- Dx / 2 / np.square(lengthscale_x)) * np.exp(-Dy/2/np.square(lengthscale_y));
    K += np.eye(Gx*Gy)*1e-9;

    K_chol = scipy.linalg.cholesky(K, lower=True)

    fouling = np.empty([N_samples, Gy, Gx])
    for i in range(N_samples):
        z = np.random.normal(0,1,Gx*Gy)
        f = np.matmul(K_chol, z)
        
        sub = np.random.rand(1)*2      
             
        f_pos = sigmoid(f - sub, scale=10)
            
        fouling[i] = np.reshape(f_pos, (Gy, Gx))         
    return fouling



#%%

Gx = utility.Gx
Gy = utility.Gy

# trajectory maps B
B = np.load('B_Dx_Dy/B.npy')

# Get distances matrix D
Dx = np.load('B_Dx_Dy/Dx.npy')
Dy = np.load('B_Dx_Dy/Dy.npy')

#%%

train_y = generate_fouling(Dx, Dy, Gx,Gy, 400000)  
valid_y = generate_fouling(Dx, Dy, Gx,Gy, 90000)
test_y = generate_fouling(Dx, Dy, Gx,Gy, 10000)

#%%

train_y = np.save('dataset/train_y.npy', train_y)
valid_y = np.save('dataset/valid_y.npy', valid_y)
test_y = np.save('dataset/test_y.npy', test_y)

#%%

