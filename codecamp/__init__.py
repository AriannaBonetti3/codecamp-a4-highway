"""Turbie functions.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate
from multiprocessing import Pool
import time


def load_resp(path_resp, t_start=0):
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
    """
    Loads the Turbie wind time series from a text file into NumPy arrays.
    
    Args:
        path_wind (str or pathlib.Path): Path to the wind data file.
        t_start (float, optional): Start time for the returned data. Defaults to 0.
    
    Returns:
        tuple: Two 1D NumPy arrays (t_wind, u_wind).
        t_wind: time [s]
        u_wind: wind speed [m/s]
    """
    path_wind = Path(path_wind)

    with path_wind.open('r') as file:  #opens the file in reading mode
        # Read header line: read the first line and splits it by tab(\t)
        headers = file.readline().strip().split('\t')  
    data = np.loadtxt(path_wind, skiprows=1)
    
    t_wind, u_wind = data[:, 0], data[:, 1]
    mask = t_wind >= t_start
    return t_wind[mask], u_wind[mask]


def plot_resp(t, u, xb, xt, xlim=(60, 660)):
    """
    Plots the response data with wind speed and displacements.
    
    Args:
        t (numpy.array): Time array [s].
        u (numpy.array): Wind speed array [m/s].
        xb (numpy.array): Blade displacement array [m].
        xt (numpy.array): Tower displacement array [m].
        xlim (tuple, optional): Limits for the x-axis. Defaults to (60, 660).
    
    Returns:
        tuple: Figure and axes of the plots.
    """
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
    """
    Loads Turbie parameters from a text file into a dictionary.
    
    Args:
        path_params (str or pathlib.Path): Path to the parameters file.
    
    Returns:
        dict: Dictionary containing Turbie parameters.
    """
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


def get_turbie_system_matrices(path_params):
    """
    Constructs and returns Turbie's mass, damping, and stiffness matrices.
    
    Args:
        path_params (str or pathlib.Path): Path to the parameters file.
    
    Returns:
        tuple: Three 2D NumPy arrays (M, C, K).
        M: Mass matrix
        C: Damping matrix
        K: Stiffness matrix
    """
    params = load_turbie_parameters(path_params)
    
    M = np.array([[params["mb"]*3, 0], [0, params["mt"]+params["mh"]+params["mn"]]])
    C = np.array([[params["c1"], -params["c1"]], [-params["c1"], params["c1"] + params["c2"]]])
    K = np.array([[params["k1"], -params["k1"]], [-params["k1"], params["k1"] + params["k2"]]])
    
    return M, C, K


def calculate_ct(u_wind, path_ct):
    """
    Calculates the mean of u_wind and interpolates the Ct value from the CT.txt file.
    
    Args:
        u_wind (numpy.array or float): Wind speed time series or a single wind speed value.
        path_ct (str or pathlib.Path): Path to the CT.txt file.
    
    Returns:
        float: Interpolated Ct value corresponding to the mean wind speed.
    """
    # Load Ct data from the file
    ct_data = np.loadtxt(path_ct, skiprows=1)  # Skip the header row
    wind_speeds = ct_data[:, 0]  # Wind speeds from the file
    ct_values = ct_data[:, 1]  # Corresponding Ct values
    
    # If u_wind is an array, calculate the mean; if it's a single value, use it directly
    if isinstance(u_wind, (int, float)):
        mean_u_wind = u_wind
    else:
        mean_u_wind = np.mean(u_wind)
    
    # Interpolate the Ct value based on the mean wind speed
    ct = np.interp(mean_u_wind, wind_speeds, ct_values)
    
    return ct



def calculate_dydt(t, y, M, C, K, ct=None, rho=None, rotor_area=None, t_wind=None, u_wind=None):
    """
    Compute the time derivative of the state vector y for Turbie system.
    
    Parameters:
    - t: Current time
    - y: Current state vector [displacements, velocities]
    - M: Mass matrix
    - C: Damping matrix
    - K: Stiffness matrix
    - ct: Coefficient for aerodynamic forcing (optional, needed for forced case)
    - rho: Air density (optional, needed for forced case)
    - rotor_area: Rotor area (optional, needed for forced case)
    - t_wind: Time series for wind speed (optional, needed for forced case)
    - u_wind: Wind speed series (optional, needed for forced case)
    
    Returns:
    - dydt: Time derivative of the state vector
    """
    N = M.shape[0]  # Degrees of freedom
    
    # Inverse of the mass matrix 
    Minv = np.linalg.inv(M)
    
    # ---- Assemble matrix A (homogeneous part) ----
    I = np.eye(N)  
    O = np.zeros((N, N)) 
    A = np.block([[O, I], [-Minv @ K, -Minv @ C]]) 
    
    # ---- Define the forcing vector F (if aerodynamic forcing is present) ----
    F = np.zeros(N)  # Initialize forcing vector
    
    if ct is not None and rho is not None and rotor_area is not None and t_wind is not None and u_wind is not None:
        # If aerodynamic forcing is enabled, calculate it
        x1_dot = y[2]  
        u = np.interp(t, t_wind, u_wind)  # Interpolate wind speed at time t
        faero = 0.5 * rho * rotor_area * ct * (u - x1_dot) * np.abs(u - x1_dot)  # Aerodynamic force
        F[0] = faero  
    
    # ---- Assemble array B (forcing term) ----
    B = np.zeros(2 * N)  # Initialize the array
    B[N:] = Minv @ F  
    
    # ---- Compute dydt ----
    dydt = A @ y + B  # First-order differential system
    return dydt


import numpy as np
import scipy.integrate

def simulate_turbie(path_wind, path_parameters, path_Ct):
    """
    Simulates the Turbie turbine response to wind forcing.

    Parameters:
        path_wind (str or pathlib.Path): Path to the wind data file (time series).
        path_parameters (str or pathlib.Path): Path to the turbine parameters file.
        path_Ct (str or pathlib.Path): Path to the turbine aerodynamic coefficient lookup table.

    Returns:
        t (np.ndarray): Time of the simulated response [s].
        u_wind (np.ndarray): Streamwise wind speed [m/s].
        x_b (np.ndarray): Displacement of the blade [m].
        x_t (np.ndarray): Displacement of the tower [m].
    """

    # Load wind and turbine parameters
    t_wind, u_wind = load_wind(path_wind)  # Time series and wind speeds
    params = load_turbie_parameters(path_parameters)  # Turbine parameters

    # Debugging: Check if data is loaded properly
    # print(f"Wind data loaded: {len(t_wind)} time steps")
    # print(f"Turbine parameters: {params}")

    # Rotor area and aerodynamic coefficient calculation
    rotor_area = np.pi * (params['Dr'] / 2)**2  # Rotor area from diameter
    ct = calculate_ct(u_wind, path_Ct)  # Aerodynamic coefficient from Ct file

    # Debugging: Check rotor area and Ct coefficient
    # print(f"Rotor area: {rotor_area}, Ct coefficient: {ct}")

    # Get turbine system matrices (M, C, K)
    M, C, K = get_turbie_system_matrices(path_parameters)

    # Debugging: Check system matrices
    # print(f"M matrix: {M.shape}, C matrix: {C.shape}, K matrix: {K.shape}")

    # Initial conditions: [0, 0, 0, 0] (displacement, velocity for blade and tower)
    y0 = [0, 0, 0, 0]

    # Time span for simulation (make sure the final time step is included)
    t0, tf, dt = t_wind[0], t_wind[-1], abs(t_wind[0] - t_wind[1])  # Ensure dt is positive
    t_eval = np.arange(t0, tf + dt, dt)  # Ensure final time step is included

    # Debugging: Check the number of time steps
    # print(f"Time span for simulation: {t0} to {tf}, dt: {dt}")
    # print(f"Time points for evaluation: {t_eval[:5]}...")

    # Solve the differential equation
    args = (M, C, K, ct, params['rho'], rotor_area, t_wind, u_wind)  
    
    res = scipy.integrate.solve_ivp(calculate_dydt, [t0, tf], y0, t_eval=t_eval, args=args, rtol=1e-4, atol=1e-7)


    # Ensure the result is converted to numpy arrays
    t = np.array(res.t)  
    y = np.array(res.y)  

    # Debugging: Check if results are returned correctly
    # print(f"Shape of t: {t.shape}, Shape of y: {y.shape}")

    # Blade and tower displacements (ensure y has at least 2 rows)
    if y.shape[0] >= 2: 
        x_b = y[0, :] - y[1, :] 
        x_t = y[1, :]  
    else:
        print("Error: State vector does not contain the expected number of rows.")
        x_b = np.zeros(t.shape)  
        x_t = np.zeros(t.shape)

    # Interpolate wind speed to match the time points of the simulation
    u_wind_interp = np.interp(t, t_wind, u_wind)

    # Return the results
    return t, u_wind_interp, x_b, x_t




def save_resp(t, u, xb, xt, path_save):
    """
    Saves the simulation results to a text file.

    Parameters:
        t (numpy array): Time steps
        u (numpy array): Wind speed
        xb (numpy array): Blade displacement
        xt (numpy array): Tower displacement
        path_save (str or Path): Path to save the file
    """
    path_save = Path(path_save)  # Ensure it's a Path object
    path_save.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Combine data into one array
    data = np.column_stack((t, u, xb, xt))

    # Save with proper formatting
    np.savetxt(
        path_save,
        data,
        delimiter="\t",
        fmt="%.3f",
        header="Time (s)\tWind Speed (m/s)\tBlade Disp. (m)\tTower Disp. (m)",
        comments=""
    )

    print(f"Simulation results saved to {path_save}")



def process_single_wind_file(wind_file, path_parameters, path_Ct):
    """
    Simulates the turbine response for a given wind file.
    
    Parameters:
    - wind_file (Path): Path to the wind file.
    - path_parameters (Path): Path to turbine parameters.
    - path_Ct (Path): Path to Ct values.
    
    Returns:
    - tuple: (mean wind speed, mean/std blade deflection, mean/std tower deflection)
    """
    t, u_wind, x_b, x_t = simulate_turbie(wind_file, path_parameters, path_Ct)
    return np.mean(u_wind), np.mean(x_b), np.std(x_b), np.mean(x_t), np.std(x_t)

def process_wind_files(data_folder, path_parameters, path_Ct):
    """
    Processes all wind files in the given folder using multiprocessing.
    
    Parameters:
    - data_folder (Path): Folder containing wind simulation files.
    - path_parameters (Path): Path to turbine parameters.
    - path_Ct (Path): Path to Ct values.

    Returns:
    - np.array: Arrays of mean wind speeds, blade deflections, and tower deflections.
    """
    wind_files = sorted(data_folder.glob("*.txt"))

    start_time = time.time()
    
    # Use multiprocessing to process files in parallel
    with Pool() as pool:
        results = pool.starmap(process_single_wind_file, [(wf, path_parameters, path_Ct) for wf in wind_files])

    # Convert results into separate NumPy arrays
    mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower = map(np.array, zip(*results))

    end_time = time.time()
    print(f"Processing time for one folder: {end_time - start_time:.2f} seconds")

    return mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower

def plot_results(mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower, title, show=False):
    """
    Plots the mean deflections of the blade and tower against wind speed with error bars.

    Parameters:
    - mean_wind_speeds (np.array): Mean wind speeds for each simulation.
    - mean_deflections_blade (np.array): Mean blade deflections.
    - std_deflections_blade (np.array): Standard deviation of blade deflections.
    - mean_deflections_tower (np.array): Mean tower deflections.
    - std_deflections_tower (np.array): Standard deviation of tower deflections.
    - title (str): Title for the plot.
    - show (bool): Whether to display the plot immediately.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(mean_wind_speeds)):
        ax.errorbar(mean_wind_speeds[i], mean_deflections_blade[i], yerr=std_deflections_blade[i], 
                    fmt='o', capsize=4, label='Blade Deflection' if i == 0 else "", 
                    color='tab:blue', markersize=6)

        ax.errorbar(mean_wind_speeds[i], mean_deflections_tower[i], yerr=std_deflections_tower[i], 
                    fmt='s', capsize=4, label='Tower Deflection' if i == 0 else "", 
                    color='tab:red', markersize=6)

    ax.set_xlabel('Mean Wind Speed [m/s]', fontsize=12)
    ax.set_ylabel('Deflection [m]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Do not block execution unless explicitly requested
    if show:
        plt.show()

def plot_combined_results(all_results, folders):
    """
    Plots the combined results of mean deflections of the blade and tower against wind speed with error bars for different turbulence intensities.
    
    Parameters:
    - all_results (list): List of tuples containing mean wind speeds, blade deflections, and tower deflections for each folder.
    - folders (list): List of folder names.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['tab:blue', 'tab:green', 'tab:orange']
    markers = ['o', 's', '^']
    labels = ['Blade Deflection', 'Tower Deflection']

    for idx, (mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower) in enumerate(all_results):
        for i in range(len(mean_wind_speeds)):
            ax.errorbar(mean_wind_speeds[i], mean_deflections_blade[i], yerr=std_deflections_blade[i], 
                        fmt=markers[0], capsize=4, label=f'{labels[0]} ({folders[idx].split("_")[-1]})' if i == 0 else "", 
                        color=colors[idx], markersize=6)

            ax.errorbar(mean_wind_speeds[i], mean_deflections_tower[i], yerr=std_deflections_tower[i], 
                        fmt=markers[1], capsize=4, label=f'{labels[1]} ({folders[idx].split("_")[-1]})' if i == 0 else "", 
                        color=colors[idx], markersize=6)

    ax.set_xlabel('Mean Wind Speed [m/s]', fontsize=12)
    ax.set_ylabel('Deflection [m]', fontsize=12)
    ax.set_title('Combined Deflections vs Wind Speed for Different Turbulence Intensities', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# def example():
#     """An example function in a package."""
#     print('This is an example!')

