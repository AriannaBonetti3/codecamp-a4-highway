[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NbRStOuB)
# Our CodeCamp project

**Team**: A4-Highway  

## PROJECT OVERVIEW:
This project analyzes wind turbines blade and tower deflections under varying wind speeds using turbulent wind time series data. It takes as inputs:
 - turbine parameters - ./data/turbie_parameters-txt
 - time series of the turbulent wind speed - 3 folders: ./data/wind_TI_0.1, 0.05 and 0.15. These 3 folders contains the turbulent wind time series
 - thrust coefficient (CT) table for each mean wind speed - ./data/CT.txt

The purpose of the code is to plot the mean and standard deviation of the tower and blade deflection.


The code performs the following key tasks:
- Loads wind data from the dataset.
- Processes turbine parameters and compute deflections
- Computes mean and standard deviation of deflection and wind speed.
- Generates plots.


## Quick-start guide

### 1. Clone the repository
```sh
git clone <repository_url>
cd <repository_directory>
```

### 2. Install packages
Ensure you have Python installed. Then, install the required packages (numpy, scipy, matplotlib ecc..):

```sh
pip install numpy scipy matplotlib
```


### 3. Run the main script
Execute the script to analyze the wind data:
```sh
python main.py
```

## 4. How the code works

The project consists of the following core components:

### **Main Script**
- **`main.py`**: The primary script that manages the entire process using the helper functions contained in **`__init__.py`** with the following steps:
  - Loads wind data from the `./data/` directory.
  - Computes mean and standard deviation of blade and tower deflections at different wind speeds.
  - Generates plots to visualize results.

### **Helper Functions**
- **`__init__.py`**: Contains the following features:

  ### Data Loading:
  - `load_resp()`: Loads turbine response data (wind speed, blade, and tower displacements).
  - `load_wind()`: Loads wind speed time series.
  - `load_turbie_parameters()`: Reads turbine parameters from a file.

  ### Simulation:
  - `get_turbie_system_matrices()`: Constructs the system's mass, damping, and stiffness matrices.
  - `calculate_ct()`: Computes the aerodynamic coefficient (Ct) from wind speed data.
  - `calculate_dydt()`: Defines the system of differential equations governing the turbine motion.
  - `simulate_turbie()`: Simulates the turbine's response to wind forcing.

  ### Results Processing & Visualization:
  - `plot_resp()`: Plots wind speed, blade, and tower displacements.
  - `save_resp()`: Saves simulation results to a file.
  - `process_wind_files()`: Processes multiple wind simulation files in parallel.
  - `plot_results()`: Visualizes mean deflections against wind speed with error bars.

    **See Diagrams.drawio for the structure**

### **Project Directories & files**
- **`./data/`**: 
    - **wind_TI_** Stores turbulent wind time series data for 3 turbulence intensities (0.1, 0.05, 0.15).
    - **CT.txt**:  thrust coeff. curve
    - **turbie_parameters**: wind turbine parameters. 
-**`./CodeCamp`** contains **`__init__.py`**.
- **`Diagrams.drawio`**: diagram of the code structure.


## Team contributions
For CodeCamp project, everybody tried to understand and begin to approach the code alone. This to have a deep understanding of the problem and a structured idea on how to solve it. 
After this, we discussed how to complete the project and together we wrote both the **`__init__.py`** and **`main.py`**. This way, everybody had a clear view of the 
code and were able to understand the solution.
This, README file as well as the other that are not afore-mentioned were also produced together as a group.