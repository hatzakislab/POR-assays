import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
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

    return slope_masked


def process_df(df, root, date, plate, plot = True, pearson_threshold = 0.95, correction_mode = "auto"):
    root = Path(root)
    path = root / date / plate / "processed"
    plot_path = path / "plots"
    slope_plot_path = plot_path / "slopes"
    full_plot_path = slope_plot_path / "full"
    corrected_plot_path = slope_plot_path / "corrected"
    auto_corrected_plot_path = slope_plot_path / "auto_corrected"
    [os.makedirs(this_path, exist_ok = True) for this_path in [path, plot_path, slope_plot_path, full_plot_path, corrected_plot_path, auto_corrected_plot_path]]


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
            fig_full, ax_full = plt.subplots()
            fig_masked, ax_masked = plt.subplots()
            fig_masked_auto, ax_masked_auto = plt.subplots()
        for column in range(1, 12 + 1):
            problematic = False
            if plot:
                color = color_code[column - 1]
            column = str(column)
            sys_row_col_df = sys_row_df[sys_row_df["column"] == column]
            slope_full, int_full, r_full, _, _ = linregress(sys_row_col_df["Time"], sys_row_col_df["data"])
            
            if r_full < pearson_threshold:
                X = sm.add_constant(sys_row_col_df["Time"])
                Y = sys_row_col_df["data"]
                model = sm.OLS(Y, X)
                results = model.fit()
                infl = results.get_influence()
                cook, _ = infl.cooks_distance
                threshold = otsu_threshold(cook)
                
                # To avoid the "copy slice warning"-thingy from pandas
                sys_row_col_df = sys_row_col_df.copy()

                sys_row_col_df.loc[cook > threshold, "automatic include"] = False
                problematic = True

            line_x = np.array([sys_row_col_df["Time"].min(), sys_row_col_df["Time"].max()])
            
            if plot:
                ax_full.plot(sys_row_col_df["Time"], sys_row_col_df["data"], "o", color = color, label = column)
                ax_full.plot(line_x, line_x * slope_full + int_full, color = color)

                ax_full.legend(**legend_opts)


            # Manual changes
            if plot:
                slope_masked = mask_with_column(sys_row_col_df, "include", column, color, ax_masked, line_x, legend_opts)
            else:
                slope_masked = mask_with_column(sys_row_col_df, "include", column)

            #Automatic changes
            if plot:
                if problematic:
                    if system == "RS":
                        ax_problematic = ax_problematic_RS
                    elif system == "CytC":
                        ax_problematic = ax_problematic_CytC
                    slope_masked_auto = mask_with_column(sys_row_col_df, "automatic include", column,
                    color, ax_masked_auto, line_x, legend_opts, ax_problematic, f"{row}{column}")
                else:
                    slope_masked_auto = mask_with_column(sys_row_col_df, "automatic include", column,
                    color, ax_masked_auto, line_x, legend_opts)
            else:
                slope_masked_auto = mask_with_column(sys_row_col_df, "automatic include", column)

            sys_row_col_dfs.append(sys_row_col_df)
            slopes.append((system, row, column, slope_full, slope_masked, slope_masked_auto))
        
        if plot:
            ax_full.set_title(f"Row {row} uncorrected")
            ax_masked.set_title(f"Row {row} corrected")
            
            plt.figure(fig_full.number)
            # plt.figtext(0.15, 0.55, color_text)
            plt.savefig(full_plot_path / f"{system}_{row}_full.png")

            plt.figure(fig_masked.number)
            plt.savefig(corrected_plot_path / f"{system}_{row}_corrected.png")
            
            plt.figure(fig_masked_auto.number)
            plt.savefig(auto_corrected_plot_path / f"{system}_{row}_corrected.png")
            
            plt.close()
            plt.close()
    if plot:
        ax_problematic_RS.set_title(f"Autocorrected entries RS")
        ax_problematic_CytC.set_title(f"Autocorrected entries CytC")

        ax_problematic_RS.legend(**legend_opts)
        ax_problematic_CytC.legend(**legend_opts)

        plt.figure(fig_problematic_CytC.number)
        plt.savefig(slope_plot_path / f"automatic_corrections_CytC.png")
        plt.figure(fig_problematic_RS.number)
        plt.savefig(slope_plot_path / f"automatic_corrections_RS.png")

        plt.close()
    new_df = pd.concat(sys_row_col_dfs)
    for col in new_df.columns:
        df[col] = new_df[col]
    df.to_csv(path / "full_dataframe.csv")
    slopes = pd.DataFrame(slopes, columns = ["system", "row", "column", "full_slope", "corrected_slope", "auto_corrected_slope"])
    
    relative_slopes = []
    if correction_mode == "manual":
        slope_col = "corrected_slope"
    elif correction_mode == "auto":
        slope_col = "auto_corrected_slope"
    
    for (system, row), sub_df in slopes.groupby(["system", "row"]):
        control = sub_df[(sub_df["column"] == "1") | (sub_df["column"] == "12")]
        control_mean = control[slope_col].mean()
        control_diff = abs(control[slope_col].diff().iloc[-1])
        control_diff_over_mean = control_diff / control_mean
        sub_df["relative_slope"] = sub_df[slope_col] / control_mean
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
            plt.title(date+plate+' CytC', fontsize = 20)
            sns.heatmap(pivoted[[str(i) for i in range(2, 12)]], linewidth = 0.5, annot = True, fmt=".2f", robust = True,
                vmax = 1.5, vmin = 0.5, cmap='seismic' )
            fig_heat.savefig(plot_path / f"{system}_heatmap.png")
            plt.close()
    
    return relative_slopes