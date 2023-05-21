import os
from pathlib import Path

import pandas as pd

import assay_processing

# file_path = Path(r"C:\Users\despo\Desktop\POR\Results")
# curdir, folders, files = next(os.walk(file_path))
# dates = [folder for folder in folders if len(folder.split('_')) == 3]

# all_dfs = []
# for date in dates:
#     plates = next(os.walk(file_path / date))[1]
#     for plate in plates:
#         try:
#             df = assay_processing.read_data(file_path, date, plate)
#             output = assay_processing.process_df(df, file_path, date, plate, plot = False)
#             output['Plate'] = plate
#             all_dfs.append(output)
#         except FileNotFoundError:
#             pass
        
# all_dfs = pd.concat(all_dfs)
# all_dfs.to_csv(file_path / 'all_plates.csv')

file_path = "."
date = "Plate 10"
plate = "TXT"

# Read the data
df = assay_processing.read_data(file_path, date, plate)


# Make corrections
# assay_processing.mask_df(df, "RS", "A", "1", lambda t: (t > 30) & (t < 100))
'''
assay_processing.mask_df(df, "CytC", "A", "2", lambda t: (((t >= 0) & (t < 10)))) 
'''
# Process and plot
output = assay_processing.process_df(df, file_path, date, plate, plot = True)