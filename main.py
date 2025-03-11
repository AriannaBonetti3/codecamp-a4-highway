""" Main code for Assignment: Processes wind data for different turbulence intensities 
and plots the corresponding results for deflections of the blade and tower. """

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import codecamp 

def main():
    """    
    Main function that processes wind files and plots the results for different turbulence intensities.
    The function:
        1. Defines paths to data folders and parameters.
        2. Processes the wind files for each turbulence intensity folder.
        3. Plots the results for each folder individually.
        4. Optionally combines and plots all results.
    """

    folders = ["data/wind_TI_0.05", "data/wind_TI_0.1", "data/wind_TI_0.15"] # List of folders containing wind data for different turbulence intensities
    path_parameters = Path("data/turbie_parameters.txt")   # Paths to parameter files used in the wind data processing
    path_Ct = Path("data/CT.txt")

    all_results = []

    for folder in folders:
        data_folder = Path(folder)
        # Process wind files using codecamp's process_wind_files function 
        mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower = codecamp.process_wind_files(data_folder, path_parameters, path_Ct)
        all_results.append((mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower))
        # Plot the deflections versus wind speed for the current folder
        codecamp.plot_results(mean_wind_speeds, mean_deflections_blade, std_deflections_blade, mean_deflections_tower, std_deflections_tower, title=f"Deflections vs Wind Speed ({folder.split('_')[-1]})",show = False)


    # Plot combined results - (Optional)
    # codecamp.plot_combined_results(all_results, folders)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    plt.show()