"""Script for the Week 3 assignment."""
import numpy as np
from pathlib import Path
import codecamp  # Importing the local package

# Define the paths to the data files
data_folder = Path("./data")
resp_file = data_folder / "resp_12_ms_TI_0.1.txt"  
wind_file = data_folder / "wind_12_ms_TI_0.1.txt"  
params_file = data_folder / "turbie_parameters.txt"

# Load the response data
t_resp, u_resp, xb_resp, xt_resp = codecamp.load_resp(resp_file, t_start=60)

# Print or inspect the loaded response data
print("Response Data Loaded:")
print(f"Time: {t_resp[:5]} ...")
print(f"Wind Speed: {u_resp[:5]} ...")
print(f"Blade Displacement: {xb_resp[:5]} ...")
print(f"Tower Displacement: {xt_resp[:5]} ...")

# Load the wind data
t_wind, u_wind = codecamp.load_wind(wind_file, t_start = 100)

# Print or inspect the loaded wind data
print("\nWind Data Loaded:")
print(f"Time: {t_wind[:5]} ...")
print(f"Wind Speed: {u_wind[:5]} ...")


# Generate and display plot 
fig, axs = codecamp.plot_resp(t_resp, u_resp, xb_resp, xt_resp)


# Load Turbie parameters and construct system matrices
params = codecamp.load_turbie_parameters(params_file)
M, C, K = codecamp.get_turbie_system_matrices(params_file)

# Print matrices for verification
print("\nTurbie System Matrices:")
print(f"Mass Matrix (M):\n{M}")
print(f"Damping Matrix (C):\n{C}")
print(f"Stiffness Matrix (K):\n{K}")