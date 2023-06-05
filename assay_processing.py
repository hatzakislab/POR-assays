import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

os.chdir(Path(__file__).parent)

ROW_INDEXING = {chr(letter) : i for i, letter in enumerate(range(ord("A"), ord("H") + 1))}


def otsu_threshold(data):
    sorted_data = np.sort(data)
    max_variance = 0
    final_threshold = np.nan

    # Consider each data point as a threshold
    for threshold in sorted_data[:-1]:
        # Separate data into two groups
        background = data[data <= threshold]
        foreground = data[data > threshold]

        # Compute weights (proportions of the data in each group)
        weight_background = len(background) / len(data)
        weight_foreground = len(foreground) / len(data)

        mean_background = np.mean(background)
        mean_foreground = np.mean(foreground)

        # Compute between-class variance
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # Update threshold if variance is maximized
        if variance > max_variance:
            max_variance = variance
            final_threshold = threshold

    return final_threshold

def sliding_window_view(arr, window_size):
    """Provide a sliding window view over a 1D numpy array."""
    arr_shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
    arr_strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=arr_shape, strides=arr_strides)

def read_data(root, date, plate):
    root = Path(root)
    letters = [chr(i) for i in range(ord("A"), ord("H")+1)]
    dfs = []
    model_systems = ["RS", "CytC"]
    for system in model_systems:
        for letter in letters:
            file = f"{system}_{letter}.txt"
            try:
                df = pd.read_csv(root / date / plate / file,
                            encoding = "utf-16",
                            delimiter='\t', header=2, skipfooter=2, engine = "python")
            except FileNotFoundError:
                continue
            df.dropna(inplace = True, axis = "columns")
            renamer = {col : col[1:] for col in df.columns[2:]}
            df.rename(renamer, axis = "columns", inplace = True)
            df.rename(columns = lambda col_name: "Temperature" if "Temperature" in col_name else col_name, inplace = True)
            df["row"] = letter
            df["system"] = system
            df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
            dfs.append(df)
    all_df = pd.concat(dfs)
    melted_df = all_df.melt(["Time", "Temperature", "system", "row"], var_name = "column", value_name = "data")
    melted_df["include"] = True
    melted_df["automatic include"] = True
    return melted_df

def cooks_distance(slope, intercept, X, Y, n):
    residuals = Y - (X * slope + intercept)
    X_with_const = np.vstack((np.ones_like(X), X)).T
    inv = np.linalg.inv(X_with_const.T @ X_with_const)
    leverages = np.sum(X_with_const * (X_with_const @ inv), axis=1)
    p = 2
    return residuals**2 / (n * p) * (leverages / (1 - leverages) ** 2)


def query_df(df, system, row, col):
    return df.loc[(df["row"] == row) & (df["column"] == col) & (df["system"] == system)]

def mask_df(df, system, row, col, time_limit, val = False):
    df.loc[
            (df["row"] == row) &
            (df["column"] == col) & 
            (df["system"] == system) &
            time_limit(df["Time"]),
            "include" ] = val

def mask_with_column(df, include_column, column, color = None, ax = None, line_x = None, legend_opts = None, ax_problematic = None, problematic_label = None):

    masked = df[df[include_column]]
    inv_masked = df[~df[include_column]]
    slope_masked, int_masked, r_masked, _, _ = linregress(masked["Time"], masked["data"])
    if ax is not None:
        ax.plot(line_x, line_x * slope_masked + int_masked, color = color)
        ax.plot(masked["Time"], masked["data"], "o", color = color, label = column)
        ax.plot(inv_masked["Time"], inv_masked["data"], "o", color = color, markerfacecolor = "none")

        ax.legend(**legend_opts)
    
    if ax_problematic is not None:
        color = next(ax_problematic._get_lines.prop_cycler)["color"]
        ax_problematic.plot(line_x, line_x * slope_masked + int_masked, color = color)
        ax_problematic.plot(masked["Time"], masked["data"], "o", color = color, label = problematic_label)
        ax_problematic.plot(inv_masked["Time"], inv_masked["data"], "o", color = color, markerfacecolor = "none")

    return slope_masked, r_masked

def autodetect(X, Y, window_size, r2_threshold):
    x_windows = sliding_window_view(X, window_size)
    y_windows = sliding_window_view(Y, window_size)

    r2_best = 0
    out_best = None
    good_enoughs = []
    for x_sub, y_sub, start in zip(x_windows, y_windows, range(10000)):
        slope, intercept, r, _, _ = linregress(x_sub, y_sub)
        # cooks = cooks_distance(slope, intercept, X, Y, window_size)
        resids = np.abs(X * slope + intercept - Y)
        
        # threshold = np.max(cooks[start: start + window_size]) * 30
        threshold = np.max(resids[start: start + window_size]) * 5
        # mask = cooks <= threshold
        mask = resids <= threshold
        res = linregress(X[mask], Y[mask])
        r2 = res[2]**2
        # print(f"raw {r**2} cooked {r2}")
        if r2 > r2_best:
            r2_best = r2
            out_best = (res, mask, X[mask].min())
        if r2 > r2_threshold:
            good_enoughs.append((res, mask, X[mask].min()))
    
    for (res, mask, x_min) in good_enoughs:
        if res[0] > out_best[0][0]:
            out_best = (res, mask, x_min)
        # if x_min < out_best[2]:
        #     out_best = (res, mask, x_min)
        # elif (x_min == out_best[2]) and (res[2]**2 > out_best[0][2]**2):
        #     out_best = (res, mask, x_min)
        
    out_best = out_best[0], out_best[1]
    return out_best

def process_df(df, root, date, plate, plot = True, pearson_threshold = 0.95, automatic_correction = True, repeat_mapping = None):
    root = Path(root)
    path = root / date / plate / "processed"
    import shutil
    shutil.rmtree(path, ignore_errors = True)
    plot_path = path / "plots"
    slope_plot_path = plot_path / "slopes"
    # full_plot_path = slope_plot_path / "full"
    # corrected_plot_path = slope_plot_path / "corrected"
    auto_corrected_plot_path = slope_plot_path / "corrected"
    [os.makedirs(this_path, exist_ok = True) for this_path in [path, plot_path, slope_plot_path, auto_corrected_plot_path]]

    df["automatic include"] = df["include"]

    if plot:
        color_code = ['black', 'red', 'yellow', 'olive', 'pink', 'green', 'teal', 'purple',
                        'brown','blue','orange','grey']
        legend_opts = dict(ncol = 6, loc = "upper left", handlelength = 0.5, borderpad = 0.3, labelspacing = 0.5, framealpha = 0.5)
    slopes = []
    sys_row_col_dfs = []

    if plot:
        fig_problematic_RS, ax_problematic_RS = plt.subplots()
        fig_problematic_CytC, ax_problematic_CytC = plt.subplots()
    for (system, row), sys_row_df in df.groupby(["system", "row"]):
        if plot:
            # fig_full, ax_full = plt.subplots()
            # fig_masked, ax_masked = plt.subplots()
            fig_masked_auto, ax_masked_auto = plt.subplots()
        for int_column in range(1, 12 + 1):
            
            problematic = False
            if plot:
                color = color_code[int_column - 1]
            column = str(int_column)
            sys_row_col_df = sys_row_df[sys_row_df["column"] == column]
            
            do_automatic = sys_row_col_df["include"].all() & automatic_correction

            line_x = np.array([sys_row_col_df["Time"].min(), sys_row_col_df["Time"].max()])
            

            # Manual changes
            masked = sys_row_col_df[sys_row_col_df["include"]]
            slope_masked, int_masked, r_masked, _, _ = linregress(masked["Time"], masked["data"])
            if r_masked < pearson_threshold:
                problematic = True

            #Automatic changes
            if do_automatic & problematic:
                _res, mask = autodetect(sys_row_col_df["Time"].values[sys_row_col_df["include"]], sys_row_col_df["data"].values[sys_row_col_df["include"]], 8, 0.97)

                sys_row_col_df = sys_row_col_df.copy()
                sys_row_col_df.loc[sys_row_col_df["include"], "automatic include"] = mask
            
            if plot:
                if problematic or not do_automatic:
                    if system == "RS":
                        ax_problematic = ax_problematic_RS
                    elif system == "CytC":
                        ax_problematic = ax_problematic_CytC
                    slope_masked, _ = mask_with_column(sys_row_col_df, "automatic include", column,
                    color, ax_masked_auto, line_x, legend_opts, ax_problematic, f"{row}{column}")
                else:
                    slope_masked, _ = mask_with_column(sys_row_col_df, "automatic include", column,
                    color, ax_masked_auto, line_x, legend_opts)
            else:
                slope_masked, _ = mask_with_column(sys_row_col_df, "automatic include", column)

            if repeat_mapping is not None:
                if int_column == 1 or int_column == 12:
                    true_name = ""
                else:
                    true_name = repeat_mapping.iloc[ROW_INDEXING[row], int_column - 2]
            else:
                true_name = ""

            sys_row_col_dfs.append(sys_row_col_df)
            slopes.append((system, row, column, slope_masked, true_name))
        
        if plot:
            plt.figure(fig_masked_auto.number)
            plt.title(f"{plate} {system}-{row}{column}")
            plt.savefig(auto_corrected_plot_path / f"{system}_{row}_corrected.png")
            plt.close()
        
    if plot:
        ax_problematic_RS.set_title(f"Changes RS {plate}")
        ax_problematic_CytC.set_title(f"Changes CytC {plate}")

        ax_problematic_RS.legend(**legend_opts)
        ax_problematic_CytC.legend(**legend_opts)

        plt.figure(fig_problematic_CytC.number)
        plt.savefig(slope_plot_path / f"Changes_CytC.png")
        plt.figure(fig_problematic_RS.number)
        plt.savefig(slope_plot_path / f"Changes_RS.png")

        plt.close()
        plt.close()
    new_df = pd.concat(sys_row_col_dfs)
    for col in new_df.columns:
        df[col] = new_df[col]
    df.to_csv(path / "full_dataframe.csv")
    # slopes = pd.DataFrame(slopes, columns = ["system", "row", "column", "full_slope", "corrected_slope", "auto_corrected_slope", "automatic correction"])
    slopes = pd.DataFrame(slopes, columns = ["system", "row", "column", "corrected_slope", "true_name"])
    
    relative_slopes = []
    # if correction_mode == "manual":
    #     slope_col = "corrected_slope"
    # elif correction_mode == "auto":
    #     slope_col = "auto_corrected_slope"
    
    for (system, row), sub_df in slopes.groupby(["system", "row"]):
        # slopes = np.where(sub_df["automatic correction"], sub_df["auto_corrected_slope"], sub_df["corrected_slope"])
        slopes = sub_df["corrected_slope"].values
        control = slopes[(sub_df["column"] == "1") | (sub_df["column"] == "12")]
        control_mean = control.mean()
        control_diff = abs(np.diff(control)[-1])
        control_diff_over_mean = control_diff / control_mean
        sub_df["relative_slope"] = slopes / control_mean
        sub_df["diff_over_mean"] = control_diff_over_mean
        relative_slopes.append(sub_df)

    relative_slopes = pd.concat(relative_slopes)
    relative_slopes.to_csv(path / "relative_slopes.csv")
    
    for system in ["RS", "CytC"]:
        pivoted = relative_slopes[relative_slopes["system"] == system].pivot(index = "row", columns = "column", values = ["relative_slope"])
        pivoted.columns = pivoted.columns.droplevel(0)
        if pivoted.empty:
            continue
        pivoted = pivoted[[str(i) for i in range(1, 13)]]
        diff_over_mean = relative_slopes[relative_slopes["system"] == system].groupby("row")["diff_over_mean"].max()
        pivoted["diff_over_mean"] = diff_over_mean
        pivoted.to_excel(path / f"{system}_relative_slopes.xlsx")

        if plot:
            fig_heat = plt.figure()
            plt.title(f"{date}-{plate}-{system}", fontsize = 20)
            sns.heatmap(pivoted[[str(i) for i in range(2, 12)]], linewidth = 0.5, annot = True, fmt=".2f", robust = True,
                vmax = 1.5, vmin = 0.5, cmap='seismic' )
            fig_heat.savefig(plot_path / f"{system}_heatmap.png")
            plt.close()
    
    return relative_slopes

def batch(file_path, corrections, plot = True, plates_filter = None, pearson_threshold = 0.95):
    file_path = Path(file_path)
    with open(file_path / "corrections.txt", "w") as file:
        json.dump(corrections, file, indent = 2)

    curdir, folders, files = next(os.walk(file_path))
    dates = [folder for folder in folders if len(folder.split('_')) == 3]

    all_dfs = []
    for date in dates:
        plates = next(os.walk(file_path / date))[1]
        for plate in plates:
            if (plates_filter is not None) and (plate not in plates_filter):
                continue
            try:
                df = read_data(file_path, date, plate)
            except FileNotFoundError:
                pass
            try:
                plate_corrections = corrections[f"{date}-{plate}"]
                for (sysrowcol, lambdastr) in plate_corrections.items():
                    sys, rowcol = sysrowcol.split("-")
                    row = rowcol[0]
                    col = rowcol[1:]
                    the_lambda = eval(lambdastr)
                    mask_df(df, sys, row, col, the_lambda)
            except KeyError:
                pass
            
            try:
                repeat_mapping = pd.read_excel(file_path / date / plate / "repeat_mapping.xlsx", index_col = 0)
                repeat_mapping.fillna("", inplace = True)
            except FileNotFoundError:
                repeat_mapping = None

            output = process_df(
                df,
                file_path,
                date,
                plate,
                plot = plot,
                pearson_threshold = pearson_threshold,
                repeat_mapping = repeat_mapping
            )
            output['Plate'] = plate
            all_dfs.append(output)
            
    all_dfs = pd.concat(all_dfs)
    all_dfs.to_csv(file_path / 'all_plates.csv')

    # Fix repeats
    plates = all_dfs.loc[all_dfs["true_name"] != "", "true_name"].apply(lambda x: f"Plate {x.split('-')[0]}")
    rows = all_dfs.loc[all_dfs["true_name"] != "", "true_name"].apply(lambda x: f"{x.split('-')[1][0]}")
    cols = all_dfs.loc[all_dfs["true_name"] != "", "true_name"].apply(lambda x: f"{x.split('-')[1][1:]}")
    systems = all_dfs.loc[all_dfs["true_name"] != "", "system"]
    for plate, row, column, system in zip(plates, rows, cols, systems):
        all_dfs = all_dfs[~(
            (all_dfs["system"] == system) &
            (all_dfs["row"] == row) &
            (all_dfs["column"] == column) &
            (all_dfs["Plate"] == plate)
        )]
    all_dfs.loc[all_dfs["true_name"] != "", "Plate"] = plates
    all_dfs.loc[all_dfs["true_name"] != "", "row"] = rows
    all_dfs.loc[all_dfs["true_name"] != "", "column"] = cols
    all_dfs.drop(columns = "true_name", inplace = True)

    all_dfs.to_csv(file_path / 'all_plates_repeats_fixed.csv')

    all_dfs = all_dfs[(all_dfs["column"] != "1") & (all_dfs["column"] != "12")]

    system_pivoted = all_dfs.pivot_table(index = ["Plate", "row", "column"], columns = ["system"], values = ["relative_slope"])
    system_pivoted.reset_index(inplace = True)
    system_pivoted.columns = ["_".join(col) if col[-1] != '' else col[0] for col in system_pivoted.columns]
    

    try:
        master = load_smiles(file_path)
        system_pivoted = system_pivoted.merge(master, how = "right", left_on = ["Plate", "row", "column"], right_on = ["Plate", "row", "column"])
    except FileNotFoundError:
        pass

    system_pivoted.to_csv(file_path / 'pivoted.csv')

    return system_pivoted

def load_smiles(file_path):
    file_path = Path(file_path)
    fda = pd.read_excel(file_path / "FDA Plates 1-13.xlsx")
    inf = pd.read_excel(file_path / "Infinisee Plate 16.xlsx")
    ena = pd.read_excel(file_path / "Enamine 6M Plates 14-15.xlsx")

    fda = fda[["Catalog ID", "Plate_ID", "Well", "Smiles"]]
    inf = inf[["Catalog_ID", "Plate_ID", "Well", "Smile"]]
    ena = ena[["Catalog_ID", "Plate_ID", "Well", "Smile"]]
    fda.rename(columns = {"Catalog ID": "Catalog_ID"}, inplace = True)
    inf.rename(columns = {"Smile": "Smiles"}, inplace = True)
    ena.rename(columns = {"Smile": "Smiles"}, inplace = True)

    fda["Plate"] = fda["Plate_ID"].str.split("-", expand = True).iloc[:, -1].astype(int)
    fda["Plate"] = "Plate " + fda["Plate"].astype(str)
    fda["row"] = fda["Well"].str.get(0)
    fda["column"] = fda["Well"].str.slice(1).astype(int).astype(str)


    inf["Plate"] = inf["Plate_ID"].str.split("-", expand = True).iloc[:, -1].astype(int)
    inf["Plate"] = "Plate " + (inf["Plate"] - 3 + 16).astype(str)
    inf["row"] = inf["Well"].str.get(0)
    inf["column"] = inf["Well"].str.slice(1).astype(int).astype(str)

    ena["Plate"] = ena["Plate_ID"].str.split("-", expand = True).iloc[:, -1].astype(int)
    ena["Plate"] = "Plate " + (ena["Plate"] - 1 + 14).astype(str)
    ena["row"] = ena["Well"].str.get(0)
    ena["column"] = ena["Well"].str.slice(1).astype(int).astype(str)

    master = pd.concat([fda, ena, inf]).drop(columns = ["Plate_ID", "Well"])
    return master

def plot_final(final_df, file_path):
    system_values = ["relative_slope_RS", "relative_slope_CytC"]
    os.makedirs(file_path / "heatmaps", exist_ok = True)
    for plate, sub_df in final_df.groupby("Plate"):
        for value in system_values:
            system = value.split("_")[-1]
            pivoted = sub_df.pivot(index = "row", columns = "column", values = [value])
            pivoted.columns = [col[-1] for col in pivoted.columns]
            try:
                pivoted = pivoted[sorted(pivoted.columns, key = lambda x: int(x))]
            except KeyError:
                continue
            sns.heatmap(pivoted, linewidth = 0.5, annot = True, fmt=".2f", robust = True,
                    vmax = 1.5, vmin = 0.5, cmap='seismic')
            plt.title(f"{plate}-{system}", fontsize = 20)
            plt.savefig(file_path / "heatmaps" / f"{plate}_{system}.png")
            plt.close()

    
    

