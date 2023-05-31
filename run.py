import os
from pathlib import Path

import pandas as pd

import assay_processing
'''
corrections = {
    "2023_03_29-Plate 1":
    {
        "CytC-E9": 'lambda t: ~((t > 50) & (t < 100))'
    },
    
    "2023_03_29-Plate 1":
    {
        "CytC-E9": 'lambda t: ~((t > 50) & (t < 100))'
        "CytC-E9": 'lambda t: ~((t > 50) & (t < 100))'
        "CytC-E9": 'lambda t: ~((t > 50) & (t < 100))'
    },
}
'''

corrections = {}




# file_path = Path(r"C:\Users\andre\Downloads\Results (R268W)")
file_path = Path(r"/Users/amfaber/Downloads/Results (R268W)")

df = assay_processing.batch(file_path, corrections)
assay_processing.plot_final(df, file_path)
print(df)
