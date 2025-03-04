"""Script for the Week 4 assignment."""
import codecamp
import numpy as np
from pathlib import Path

# Define the paths to the data files
data_folder = Path("./data")
resp_file = data_folder / "resp_12_ms_TI_0.1.txt"  
wind_file = data_folder / "wind_12_ms_TI_0.1.txt"  
params_file = data_folder / "turbie_parameters.txt"

# Load the wind data
t_wind, u_wind = codecamp.load_wind(wind_file)

# Path to CT.txt file
path_ct = './data/CT.txt'  # Replace with your actual file path

# Calculate Ct using the loaded wind speeds
ct = codecamp.calculate_ct(u_wind, path_ct)
print(f"Calculated Ct: {ct}")

# Load Turbie parameters and system matrices
params = codecamp.load_turbie_parameters(params_file)
M, C, K = codecamp.get_turbie_system_matrices(params_file)

# Example usage for the dydt function
t = 1  # Time step for the calculation
y = [1, 2, 3, 4]  # Example state vector: [displacement_1, displacement_2, velocity_1, velocity_2]

# Define necessary parameters for forced response calculation
kwargs = {
    "rho": params["rho"],  # Air density
    "ct": ct,  # Ct coefficient calculated above
    "rotor_area": params["Dr"]**2 * np.pi / 4,  # Rotor area (using the diameter)
    "t_wind": t_wind,  # Time array from wind data
    "u_wind": u_wind   # Wind speed array
}

# Call the dydt function to get the differential of the system with forcing
rhs = codecamp.calculate_dydt(t, y, M, C, K, **kwargs)

# Print the result to check
print("Calculated dydt:", rhs)


# Simulate the Turbie system response using the simulate_turbie function
t_sim, u_sim, xb, xt = codecamp.simulate_turbie(wind_file, params_file, path_ct)

# Print the first few values to check simulation results
print("Time steps of simulation:", t_sim[:5])
print("Simulated Wind speed:", u_sim[:5])
print("Simulated Blade displacement:", xb[:5])
print("Simulated Tower displacement:", xt[:5])


# Define save path in a new subfolder 'resp/'
save_path = Path("resp/test_resp.txt")

# Save the results
codecamp.save_resp(t_sim, u_sim, xb, xt, save_path)

data1 = np.loadtxt("resp/test_resp.txt",skiprows = 1)
data2 = np.loadtxt("data/resp_12_ms_TI_0.1.txt",skiprows = 1)

np.testing.assert_allclose(data1, data2, atol=1e-2)