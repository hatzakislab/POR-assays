import pandas as pd
from pathlib import Path
from functools import reduce
from assay_processing import load_smiles

file_path = Path(r"data")

variants = ["P228L", "R268W", "WT", "Sorghum",]

dfs = []

for variant in variants:
    df = pd.read_csv(file_path / variant / "pivoted.csv", index_col = 0)
    df["column"] = df["column"].astype(str)
    renamer = {col: col + f"_{variant}" for col in df.columns if not col in ["Plate", "row", "column"]}
    df.rename(columns = renamer, inplace = True)
    dfs.append(df)



full_df = reduce(lambda left, right: pd.merge(left, right, on = ["Plate", "row", "column"], how = "outer"), dfs)
catalog = load_smiles(".")
full_df = full_df.merge(catalog, on = ["Plate", "row", "column"], how = "left")

print(full_df)
