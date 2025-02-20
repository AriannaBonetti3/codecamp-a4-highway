"""Turbie functions.
"""
import numpy as np
from pathlib import Path


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
    
    return t[mask], u[mask], xb[mask], xt[mask], headers #Return filtered data

def load_wind(path_wind, t_start=0):
    """Loads the Turbie wind time series from a text file into NumPy arrays."""
    path_wind = Path(path_wind)

    with path_wind.open('r') as file:  #opens the file in reading mode
        # Read header line: read the first line and splits it by tab(\t)
        headers = file.readline().strip().split('\t')  
    data = np.loadtxt(path_wind, skiprows=1)
    
    t_wind, u_wind = data[:, 0], data[:, 1]
    mask = t_wind >= t_start
    return t_wind[mask], u_wind[mask], headers


# def example():
#     """An example function in a package."""
#     print('This is an example!')

