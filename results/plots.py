#!/usr/bin/env python3

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# ACADEMIC PLOT SETTINGS
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,        # High resolution for print
    "savefig.bbox": "tight"
})

RESULTS_FOLDER = "results" if os.path.exists("results") else "."
OUTPUT_FOLDER = "academic_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# DATA LOADING
# -------------------------
def load_all_data(folder):
    files = glob.glob(os.path.join(folder, "*.json"))
    all_data = []
    for f in files:
        with open(f, "r") as fp:
            try:
                data = json.load(fp)
                if isinstance(data, list): all_data.extend(data)
                else: all_data.append(data)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
    return pd.DataFrame(all_data)

# -------------------------
# CATEGORIZATION LOGIC
# -------------------------
def get_bit_category(row):
    """Categorizes bits based on their mathematical function."""
    dtype = str(row['data_type']).lower()
    bit_idx = str(row.get('bit_idx', '')).lower()
    
    if "exponent" in bit_idx or "sign" in bit_idx:
        return "Exponent-Sign"
    elif "mantissa" in bit_idx:
        return "Mantissa"
    elif "int" in dtype:
        return "Integer-Bits"
    return "General"

# -------------------------
# PLOTTING ENGINE
# -------------------------
def plot_academic_graphs(df):
    if df.empty:
        print("DataFrame is empty. Check your JSON files.")
        return

    df['category'] = df.apply(get_bit_category, axis=1)
    
    # Grouping by Model and Dataset to create distinct figures
    for (model, dataset), group in df.groupby(["model_name", "dataset"]):
        
        # Create sub-plots for each category (Mantissa vs Exponent vs Int)
        categories = group['category'].unique()
        
        for cat in categories:
            cat_df = group[group['category'] == cat]
            
            plt.figure(figsize=(7, 5))

            markers = ['o', 's', '^', 'D', 'x', 'v', '<', '>']
            
            for i, ((dtype, bit), line_df) in enumerate(cat_df.groupby(["data_type", "bit_idx"])):
                line_df = line_df.sort_values("ber")
                
                # Plotting line
                plt.plot(line_df["ber"], line_df["accuracy_corrupted"], 
                         label=f"{dtype} (Bit {bit})", 
                         marker=markers[i % len(markers)], 
                         markersize=5, linewidth=1.5)

            # Axis Styling
            plt.xscale("log")
            plt.ylim(-5, 105) # Standardized Y-axis for academic comparison
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            
            plt.xlabel("Bit Error Rate (BER)")
            plt.ylabel("Accuracy After Corruption (%)")
            plt.title(f"{model} - {dataset} : {cat} Analysis")
            
            # Place legend outside if too many lines, else inside
            if len(cat_df.groupby(["data_type", "bit_idx"])) > 5:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.legend(loc='best')

            # Save with high quality
            filename = f"{model}_{dataset}_{cat.replace('-', '_')}.pdf" 
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            plt.savefig(save_path, format='pdf')
            plt.savefig(save_path.replace(".pdf", ".png"), format='png') 
            plt.close()
            print(f"Generated academic plot: {save_path}")

if __name__ == "__main__":
    data = load_all_data(RESULTS_FOLDER)
    plot_academic_graphs(data)#!/usr/bin/env python3

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# ACADEMIC PLOT SETTINGS
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

RESULTS_FOLDER = "results" if os.path.exists("results") else "."
OUTPUT_FOLDER = "academic_plots_separated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# DATA LOADING
# -------------------------
def load_all_data(folder):
    files = glob.glob(os.path.join(folder, "*.json"))
    all_data = []
    for f in files:
        with open(f, "r") as fp:
            try:
                data = json.load(fp)
                if isinstance(data, list): all_data.extend(data)
                else: all_data.append(data)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
    return pd.DataFrame(all_data)

# -------------------------
# CATEGORIZATION LOGIC
# -------------------------
def get_bit_category(row):
    dtype = str(row['data_type']).lower()
    bit_idx = str(row.get('bit_idx', '')).lower()
    
    if "exponent" in bit_idx or "sign" in bit_idx:
        return "Exponent-Sign"
    elif "mantissa" in bit_idx:
        return "Mantissa"
    elif "int" in dtype:
        return "Integer-Bits"
    return "General"

# -------------------------
# PLOTTING ENGINE
# -------------------------
def plot_academic_graphs(df):
    if df.empty:
        print("DataFrame is empty. Check your JSON files.")
        return

    df['category'] = df.apply(get_bit_category, axis=1)
    
    # We now group by data_type (fp16 vs fp32) as well
    group_cols = ["model_name", "dataset", "category", "data_type"]
    
    for (model, dataset, cat, dtype), sub_df in df.groupby(group_cols):
        
        plt.figure(figsize=(7, 5))
        markers = ['o', 's', '^', 'D', 'x', 'v', '<', '>']
        
        # Plot each bit index for this specific format
        for i, (bit, line_df) in enumerate(sub_df.groupby("bit_idx")):
            line_df = line_df.sort_values("ber")
            
            plt.plot(line_df["ber"], line_df["accuracy_corrupted"], 
                     label=f"{dtype} (Bit {bit})", 
                     marker=markers[i % len(markers)], 
                     markersize=5, linewidth=1.5)

        # Axis Styling
        plt.xscale("log")
        plt.ylim(-5, 105) 
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        
        plt.xlabel("Bit Error Rate (BER)")
        plt.ylabel("Accuracy After Corruption (%)")
        plt.title(f"{model} - {dataset} : {dtype} {cat}")
        
        # Legend
        plt.legend(loc='best')

# Clean filename formatting for PNG
        clean_dtype = str(dtype).replace(".", "_")
        clean_cat = cat.replace("-", "_")
        filename = f"{model}_{dataset}_{clean_dtype}_{clean_cat}.png" 
        
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        plt.savefig(save_path) # Defaulting to PNG based on filename extension
        plt.close()
        print(f"Generated PNG plot: {save_path}")

if __name__ == "__main__":
    data = load_all_data(RESULTS_FOLDER)
    plot_academic_graphs(data)