import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

os.chdir(Path(__file__).parent)

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
            df = pd.read_csv(root / date / plate / file,
                        encoding = "utf-16",
                        delimiter='\t', header=2, skipfooter=2, engine = "python")
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
        # print("hello")
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
    for x_sub, y_sub in zip(x_windows, y_windows):
        slope, intercept, r, _, _ = linregress(x_sub, y_sub)
        cooks = cooks_distance(slope, intercept, X, Y, window_size)
        
        threshold = np.sort(cooks)[window_size - 1] * 20
        mask = cooks <= threshold
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

def process_df(df, root, date, plate, plot = True, pearson_threshold = 0.95, automatic_correction = True):
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
        legend_opts = dict(ncol = 6, loc = "lower right", handlelength = 0.5, borderpad = 0.3, labelspacing = 0.5, framealpha = 0.5)
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
        for column in range(1, 12 + 1):
            problematic = False
            if plot:
                color = color_code[column - 1]
            column = str(column)
            sys_row_col_df = sys_row_df[sys_row_df["column"] == column]
            
            # slope_full, int_full, r_full, _, _ = linregress(sys_row_col_df["Time"], sys_row_col_df["data"])
            
            do_automatic = sys_row_col_df["include"].all() & automatic_correction

            line_x = np.array([sys_row_col_df["Time"].min(), sys_row_col_df["Time"].max()])
            
            # if plot:
            #     ax_full.plot(sys_row_col_df["Time"], sys_row_col_df["data"], "o", color = color, label = column)
            #     ax_full.plot(line_x, line_x * slope_full + int_full, color = color)

            #     ax_full.legend(**legend_opts)


            # Manual changes
            masked = sys_row_col_df[sys_row_col_df["include"]]
            slope_masked, int_masked, r_masked, _, _ = linregress(masked["Time"], masked["data"])
            if r_masked < pearson_threshold:
                problematic = True
            # if plot:
            #     slope_masked, r_masked = mask_with_column(sys_row_col_df, "include", column, color, ax_masked, line_x, legend_opts)
            # else:
            #     slope_masked, r_masked = mask_with_column(sys_row_col_df, "include", column)

            #Automatic changes
            if do_automatic & problematic:
                _, mask = autodetect(sys_row_col_df["Time"].values[sys_row_col_df["include"]], sys_row_col_df["data"].values[sys_row_col_df["include"]], 8, 0.97)
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

            sys_row_col_dfs.append(sys_row_col_df)
            # slopes.append((system, row, column, slope_full, slope_masked, slope_masked_auto, do_automatic))
            slopes.append((system, row, column, slope_masked))
        
        if plot:
            # ax_full.set_title(f"Row {row} uncorrected")
            # ax_masked.set_title(f"Row {row} corrected")
            
            # plt.figure(fig_full.number)
            # plt.figtext(0.15, 0.55, color_text)
            # plt.savefig(full_plot_path / f"{system}_{row}_full.png")

            # plt.figure(fig_masked.number)
            # plt.savefig(corrected_plot_path / f"{system}_{row}_corrected.png")
            
            plt.figure(fig_masked_auto.number)
            plt.savefig(auto_corrected_plot_path / f"{system}_{row}_corrected.png")
            
            # plt.close()
            plt.close()
    if plot:
        ax_problematic_RS.set_title(f"Changes RS")
        ax_problematic_CytC.set_title(f"Changes CytC")

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
    slopes = pd.DataFrame(slopes, columns = ["system", "row", "column", "corrected_slope"])
    
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

def batch(file_path, corrections, plot = True, plates_filter = None):
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
            
            output = process_df(df, file_path, date, plate, plot = plot)
            output['Plate'] = plate
            all_dfs.append(output)
            
    all_dfs = pd.concat(all_dfs)
    all_dfs.to_csv(file_path / 'all_plates.csv')
    return all_dfs