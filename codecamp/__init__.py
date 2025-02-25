"""Turbie functions.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_resp(path_resp, t_start=60):
    """Loads the Turbie response from a text file into NumPy arrays.
    
    Args:
        path_resp (str or pathlib.Path): Path to the response file.
        t_start (float, optional): Start time for the returned data. Defaults to 0.
    
    Returns:
        tuple: Four 1D NumPy arrays (t, u, xb, xt).
        t: time [s]
        u: wind speed [m/s]
        xb: relative displacement of the baldes [m]
        xt displacement of the tower [m]
    """
    path_resp = Path(path_resp)  # Ensure it's a Path object
    
    with path_resp.open('r') as file:  #opens the file in reading mode
        # Read header line: read the first line and splits it by tab(\t)
        headers = file.readline().strip().split('\t')  
    
    data = np.loadtxt(path_resp, skiprows=1)  # Skip the header row
    
    #Extract relevant columns 
    t, u, xb, xt = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    mask = t >= t_start #Appluy time filtering
    
    return t[mask], u[mask], xb[mask], xt[mask] #Return filtered data

def load_wind(path_wind, t_start=0):
    """Loads the Turbie wind time series from a text file into NumPy arrays."""
    path_wind = Path(path_wind)

    with path_wind.open('r') as file:  #opens the file in reading mode
        # Read header line: read the first line and splits it by tab(\t)
        headers = file.readline().strip().split('\t')  
    data = np.loadtxt(path_wind, skiprows=1)
    
    t_wind, u_wind = data[:, 0], data[:, 1]
    mask = t_wind >= t_start
    return t_wind[mask], u_wind[mask]


def plot_resp(t, u, xb, xt, xlim=(60, 660)):
    """Plots the response data with wind speed and displacements."""
    fig, axs = plt.subplots(2, 1, figsize=(9, 4))
    
    # Plot wind speed
    axs[0].plot(t, u, label='Wind Speed [m/s]', color='tab:blue')
    axs[0].set_ylabel("Wind Speed [m/s]")
    axs[0].set_xlim(xlim)
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot blade and tower displacements
    axs[1].plot(t, xb, label='Blade Displacement [m]', color='tab:green')
    axs[1].plot(t, xt, label='Tower Displacement [m]', color='tab:red')
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Displacement [m]")
    axs[1].set_xlim(xlim)
    axs[1].legend()
    axs[1].grid(True)
    
    fig.tight_layout()
    plt.show()
    
    return fig, (axs[0],axs[1])


def load_turbie_parameters(path_params):
    """Loads Turbie parameters from a text file into a dictionary."""
    path_params = Path(path_params)
    
    data = np.loadtxt(path_params, comments='%')
    
    parameters = {
        "mb": data[0],
        "mn": data[1],
        "mh": data[2],
        "mt": data[3],
        "c1": data[4],
        "c2": data[5],
        "k1": data[6],
        "k2": data[7],
        "fb": data[8],
        "ft": data[9],
        "drb": data[10],
        "drt": data[11],
        "Dr": data[12],
        "rho": data[13]
    }
    
    return parameters


######### FANCY CODE FOR TURB PARAMETERS #######


# def load_turbie_parameters(path_params):
#     """Loads Turbie parameters dynamically from a text file into a dictionary."""
#     path_params = Path(path_params)
    
#     parameters = {}
#     with open(path_params, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line.startswith('%') or not line:
#                 continue  # Skip comments and empty lines
            
#             parts = line.split('%')  # Split value from comment
#             if len(parts) < 2:
#                 continue  # Skip lines without proper format
            
#             value = float(parts[0].strip())  # Extract and convert value
#             name = parts[1].strip().split('[')[0].strip()  # Extract variable name
            
#             parameters[name] = value
    
#     return parameters

######################################################


def get_turbie_system_matrices(path_params):
    """Constructs and returns Turbie's mass, damping, and stiffness matrices."""
    params = load_turbie_parameters(path_params)
    
    M = np.array([[params["mb"]*3, 0], [0, params["mt"]+params["mh"]+params["mn"]]])
    C = np.array([[params["c1"], -params["c1"]], [-params["c1"], params["c1"] + params["c2"]]])
    K = np.array([[params["k1"], -params["k1"]], [-params["k1"], params["k1"] + params["k2"]]])
    
    return M, C, K

# def example():
#     """An example function in a package."""
#     print('This is an example!')

