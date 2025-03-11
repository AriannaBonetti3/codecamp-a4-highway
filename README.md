[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NbRStOuB)
# Our CodeCamp project

**Team**: A4-Highway  

## PROJECT OVERVIEW:
This project analyzes wind turbines blade and tower deflections under varying wind speeds using turbulent wind time series data. It takes as inputs:
 - turbine parameters
 -  time series of the turbulent wind speed
 - thrust coefficient (CT) table for each mean wind speed

This, to plot the mean and standard deviation of the tower and blade deflection.


The code performs the following key tasks:
- Loads wind data from the dataset.
- Processes turbine parameters.
- Runs simulations to analyze deflections.
- Computes statistical mean and standard deviation.
- Generates plots.


## Quick-start guide

### 1. Clone the repository
```sh
git clone <repository_url>
cd <repository_directory>
```

### 2. Install dependencies
Ensure you have Python installed. Then, install the required dependencies (numpy, scipy ecc..):

```sh
pip install requirements
```


### 3. Run the main script
Execute the script to analyze the wind data:
```sh
python main.py
```

## How the code works

The project consists of the following core components:

### **Main Script**
- **`main.py`**: The primary script that manages the entire process using the helper functions contained in **`__init__.py`**:
  - Loads wind data from the `./data/` directory.
  - Computes mean and standard deviation of blade and tower deflections at different wind speeds.
  - Generates plots to visualize the results.

### **Helper Functions**
- **`__init__.py`**: Contains functions for:
  - Data loading and preprocessing, for example:

    ```sh
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
        ...
    ```

  - Statistical calculations (mean, standard deviation) from the time series files.
  - Plot the mean and standard deviation of tower and blade displacement.

    **See Diagrams.drawio for the structure**

### **Project Directories**
- **`./data/`**: 
    - **wind_TI_** Stores turbulent wind time series data for 3 turbulence intensities (0.1, 0.05, 0.15).
    - **CT.txt**:  thrust coeff. curve
    - **turbie_parameters**: wind turbine parameters. 

## Team contributions
For CodeCamp project, everybody tried to understand and begin to approach the code alone. This to have a deep understanding of the problem and a structured idea on how to solve it. 
After this, we discussed with each other how to complete the project and together we wrote both the **`__init__.py`** and **`main.py`**. This way, everybody had a clear view of the 
code and were able to understand the solution.
This, README file as well as the other that are not afore-mentioned were also produced together as a group.