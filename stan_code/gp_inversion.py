import stan
import nest_asyncio
import numpy as np
import scipy
import time

# Apply nest_asyncio to reuse the event loop
nest_asyncio.apply()

# Stan model code for the fouling problem
fouling_code = """
data {
  int<lower=1> N;           // Number of paths (observations)
  int<lower=1> Gx;          // Grid size in x-direction
  int<lower=1> Gy;          // Grid size in y-direction
  matrix[N, Gx*Gy] paths;   // Signal paths (N paths by Gx*Gy grid points)
  vector[N] attenuation_ratios; // Observed attenuation ratios
  matrix[Gx*Gy, Gx*Gy] K_chol;  // Cholesky decomposition of covariance matrix
  real<lower=0> y_sigma;         // Noise standard deviation for Gaussian likelihood
  real<lower=0> C;               // Constant for attenuation ratio calculation
}

parameters {
  vector[Gx*Gy] z;          // Latent variable for fouling map
  real<lower=0, upper=2> sub;    // Shift for sigmoid transformation
}

model {
  // Priors
  z ~ normal(0, 1); // Prior for z
  sub ~ uniform(0, 2); // Uniform prior for sub between 0 and 2

  // Generate f using the Cholesky factor of the covariance matrix
  vector[Gx*Gy] f;
  f = K_chol * z;

  // Apply the sigmoid transformation to obtain the fouling map
  vector[Gx*Gy] fouling_map = 1 / (1 + exp(-10 * (f - sub))); 

  // Forward model for attenuation ratios with Gaussian likelihood
  for (n in 1:N) {
    real overlap = dot_product(paths[n], fouling_map);  // Integral along the path
    real overlap_ratio = exp(-overlap / C); // Calculate attenuation ratio
    attenuation_ratios[n] ~ normal(overlap_ratio, y_sigma); // Gaussian likelihood
  }
}
"""


C = 10
N = 10*4 # Number of paths (observations)

# pipe measurements
Gx = 50
Gy = 48


# Known lengthscales
lengthscale_x = 5.0  
lengthscale_y = 14.0 


# Data load
iteration = 2
Dx = np.load('dataset/Dx.npy')
Dy = np.load('dataset/Dy.npy')
test_input = np.load('dataset/test_input.npy')[:iteration]
test_y = np.load('dataset/test_y.npy')[:iteration]
B = np.load('dataset/B.npy')



K = np.exp(- Dx / 2 / np.square(lengthscale_x)) * np.exp(-Dy/2/np.square(lengthscale_y));
K += np.eye(Gx*Gy)*1e-9;
K_chol = scipy.linalg.cholesky(K, lower=True)


num_chains = 4
num_samples_per_chain = 10  
num_warmup = 10  


mean_fouling_maps = []
std_fouling_maps = []

for idx in range(iteration):
    # Prepare the data for the current sample
    fouling_data = {
        'N': N,
        'Gx': Gx,
        'Gy': Gy,
        'paths': B,
        'attenuation_ratios': test_input[idx],
        'K_chol': K_chol,
        'y_sigma': 0.01,
        'C': C
    }
    
    # Build the Stan model 
    print("Building Stan model...")
    posterior = stan.build(fouling_code, data=fouling_data)
    print("Stan model built successfully.")


    print(f"Starting sampling for sample {idx + 1}...")
    start_time = time.time()

    # Sample from the posterior distribution with new data
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples_per_chain, num_warmup=num_warmup)

    end_time = time.time()
    print(f"Sampling completed for sample {idx + 1}. Time taken: {end_time - start_time:.2f} seconds")

    # Extract the samples
    z_samples = fit["z"].T  # Shape: (num_samples * num_chains, Gx * Gy)
    sub_samples = fit["sub"].flatten()  # Shape: (num_samples * num_chains,)

    # Reconstruct the fouling maps for each sample
    fouling_maps = np.zeros((num_chains * num_samples_per_chain, Gy, Gx))
    for i in range(len(sub_samples)):
        z_sample = z_samples[i]
        f_sample = np.matmul(K_chol, z_sample)
        sub_sample = sub_samples[i]
        fouling_map = 1 / (1 + np.exp(-10 * (f_sample - sub_sample)))
        fouling_maps[i] = fouling_map.reshape(Gy, Gx)

    # Compute statistics
    mean_fouling_map = np.mean(fouling_maps, axis=0)
    std_fouling_map = np.std(fouling_maps, axis=0)
    
    # Append to lists
    mean_fouling_maps.append(mean_fouling_map)
    std_fouling_maps.append(std_fouling_map)

# Convert to numpy arrays
mean_fouling_maps = np.array(mean_fouling_maps)
std_fouling_maps = np.array(std_fouling_maps)

# Save both
np.save('results/mean_fouling_maps.npy', mean_fouling_maps)
np.save('results/std_fouling_maps.npy', std_fouling_maps)

print("Mean and std fouling maps successfully saved.")

