import numpy as np
import tool
import utility
#%%

def get_D(Gx, Gy):
    Dx = np.zeros((Gx * Gy, Gx * Gy))
    Dy = np.zeros((Gx * Gy, Gx * Gy))
    for ix in np.arange(Gx):
        for iy in np.arange(Gy):
            for jx in np.arange(Gx):
                for jy in np.arange(Gy):

                    Dx[iy * Gx + ix, jy * Gx + jx] = (jx - ix) ** 2.0
                    
                    # using sin
                    Dy[iy * Gx + ix, jy * Gx + jx] = ((Gy - 1)*np.sin(1 * np.pi * np.abs(jy - iy) / (Gy - 1))) ** 2.0 
                    

    return Dx,Dy


def get_path_length(max_order, distance_x, distance_y, pipe_dia):
    path_length = []
    pipe_circ = np.pi * pipe_dia
    for order in range(round(max_order/2)+1):
        if order == 0:
            path_length.append(np.sqrt(distance_x**2 + distance_y**2))
        else:
            path_length.append(np.sqrt(distance_x**2 + (distance_y - order * pipe_circ)**2))
            path_length.append(np.sqrt(distance_x**2 + (distance_y + order * pipe_circ)**2))
    return np.asarray(path_length)[:max_order]

#%%

pipe = utility.pipe

# pixel resolution
Gx = utility.Gx
Gy = utility.Gy

# Number of tx locations
N_tx = utility.N_tx
N_rx = utility.N_rx

# Number of helical path
N_path = 10
B = np.array([])
for i in range(N_tx):
    b, N_helical = tool.get_path_grid(laser_crd = utility.location_tx[i],
                           transducer_crd = utility.location_rx[i], 
                           pipe_rad = pipe/2/np.pi, 
                           N_path = N_path,
                           grid_size = (Gx, Gy))
    if i == 0:
        B = b
    else:
        B = np.vstack([B, b])
        
 
#%%

path_length = get_path_length(max_order=10, distance_x=50, distance_y=np.pi*15.1/4, pipe_dia=15.1)

B_tmp = B[N_path].copy()
B[N_path] = B[N_path+1]
B[N_path+1] = B_tmp

B *= (np.tile(path_length,4)/B.sum(axis=1)).reshape(-1,1)

#%% Save trajectory maps B

np.save('B_Dx_Dy/B.npy', B)

#%%
# Calculate Dx and Dy
Dx, Dy = get_D(Gx, Gy)

np.save('B_Dx_Dy/Dx.npy', Dx)
np.save('B_Dx_Dy/Dy.npy', Dy)

#%%






