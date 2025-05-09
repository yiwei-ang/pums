import os
import argparse
import numpy as np
import pandas as pd

from engine.query_generator import generate_sql
from engine.utils.snowflake_connector import SnowflakeConnector

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Prepare data from Snowflake for modeling or inference.")
parser.add_argument("--person_table", required=True, help="Snowflake person-level table (e.g., public.psam_p05)")
parser.add_argument("--housing_table", required=True, help="Snowflake housing-level table (e.g., public.psam_h05)")
parser.add_argument("--output_file", default="data.parquet", help="Output file name in working_dir/")
args = parser.parse_args()

# Example:
# python engine/data_preparation.py --person_table public.psam_p05 --housing_table public.psam_h05 --output_file working_dir/data_train.parquet
# python engine/data_preparation.py --person_table public.psam_p06 --housing_table public.psam_h06 --output_file working_dir/data.parquet

# --- Generate and run SQL ---
sf = SnowflakeConnector()
sql = generate_sql(args.person_table, args.housing_table)
df = sf.query(sql)
df = df[~df['FS'].isna()]
df = df.reset_index(drop=True)
df["FS"] = df["FS"].map({1: 1, 2: 0})


# --- Count allocation flags ---
alloc_cols = [
    col for col in df.columns
    if col.startswith("F") and col.endswith("P") and col not in ['FULFP', 'FULP', 'FINCP']
]
df["num_imputed_features"] = df[alloc_cols].eq(1).sum(axis=1)


# --- Grouping: CPLT ---
def map_cplt(val):
    if val in [1, 2]:
        return "Spouse_Household"
    elif val in [3, 4]:
        return "Partner_Household"
    else:
        return "Not_Couple"


df["CPLT_group"] = df["CPLT"].replace("b", np.nan).astype(float).apply(map_cplt)


# --- Grouping: WKEXREL ---
def map_wkexrel(val):
    if val in [1, 2, 4, 5]:
        return "Both_Worked"
    elif val in [3, 6]:
        return "Householder_Worked_Only"
    elif val in [7, 8]:
        return "Spouse_Worked_Only"
    elif val == 9:
        return "Neither_Worked"
    elif val in [10, 13]:
        return "Householder_Alone_FT"
    elif val in [11, 14]:
        return "Householder_Alone_Part"
    elif val in [12, 15]:
        return "Householder_Alone_None"
    else:
        return "No_Family"


df["WKEXREL_group"] = df["WKEXREL"].replace("bb", np.nan).astype(float).apply(map_wkexrel)

# --- One-hot encode ---
df = pd.get_dummies(df, columns=["CPLT_group", "WKEXREL_group"], dtype=int)

# --- Drop original categorical columns ---
df.drop(columns=["CPLT", "WKEXREL"], inplace=True, errors='ignore')
df = df.loc[:, ~df.columns.duplicated()]
df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

# --- Save to Parquet ---
df.to_parquet(args.output_file, index=False)
print(f"Data saved to {args.output_file}")
