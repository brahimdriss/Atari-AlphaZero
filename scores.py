import os
import numpy as np

# Define the path to your main folder
main_folder = 'logs/dist'


def extract_and_group_values(subfolder_path):
    uct_values = []
    ts_values = []
    ucbts_values = []
    for root, _, files in os.walk(subfolder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Assuming the log files contain a single value per line
            with open(file_path, 'r') as file:
                value = file.readline().strip()
                if '_TS' in file_name:
                    ts_values.append(float(value))  # Convert to float or appropriate data type
                elif '_uct' in file_name:
                    uct_values.append(float(value))
                elif '_UCB-TS' in file_name:
                    ucbts_values.append(float(value))
    return ts_values, uct_values, ucbts_values

# Iterate through subfolders in the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        ts_values, uct_values, ucbts_values = extract_and_group_values(subfolder_path)
        print(f"Values in {subfolder} \n TS: {ts_values} - Mean {np.mean(ts_values)}- :CI {np.std(ts_values, ddof=1)*1.96/np.sqrt(10)}, \
        \n UCBTS: {ucbts_values} - Mean {np.mean(ucbts_values)}- CI {np.std(ucbts_values, ddof=1)*1.96/np.sqrt(10)} , \
        \n uct: {uct_values} - Mean {np.mean(uct_values)}- CI {np.std(uct_values, ddof=1)*1.96/np.sqrt(10)}")
        print("___________")