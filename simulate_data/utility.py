import numpy as np
import math

# pipe measurements
diameter = 15.1
pipe = math.ceil(diameter * np.pi) # circumference of pipe 
pipe_length = 50
shift_y = 5

location_tx = np.array([
                        [0, 0 + shift_y], # RAUS2 
                        [0, 0 + shift_y], # RAUS2 
                        
                        [0, pipe/2 + shift_y], # RAUS3 
                        [0, pipe/2 + shift_y]  # RAUS3 
                       ])

location_rx = np.array([
                        [50, 0.25*pipe + shift_y], # RAUS4
                        [50, 0.75*pipe + shift_y], # RAUS5
                        
                        [50, 0.25*pipe + shift_y], # RAUS4
                        [50, 0.75*pipe + shift_y] # RAUS5
                       ])

# pixel resolution
Gx = int(pipe_length)
Gy = int(pipe)

# Number of tx locations
N_tx = location_tx.shape[0]
N_rx = location_rx.shape[0]

