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

corrections = {
    "P228L": {
        "2023_03_29-Plate 1": {
            "CytC-A2": 'lambda t: ~((t < 40) | ((t > 75) & (t < 125)))'
        }
    },
    "R268W": {
    },
    "WT": {
    },
    "Sorghum": {
    },
}




file_path = Path(r"data")
# file_path = Path(r"/Users/amfaber/Downloads/Results (R268W)")


for key, correct in corrections.items():
    df = assay_processing.batch(file_path / key, correct)
    assay_processing.plot_final(df, file_path / key)
