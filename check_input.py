#!/usr/bin/env python3
import sys
import argparse

import pandas as pd
import numpy as np

def analyze_csv_stats(file_path):
    """
    Analyzes a CSV file to find max, min values and NaN locations with line numbers.
    Line numbers are 1-based and refer to the row position in the DataFrame
    (i.e. the first data row is line 1).
    """
    # Read CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("The CSV is empty.")
        return

    # Only consider numeric columns for max/min reporting
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n=== Analysis of '{file_path}' ===")
    print(f"Total rows (data only): {len(df)}")
    print(f"Numeric columns found: {numeric_cols}\n")

    # MAX
    print("Maximum values per numeric column:")
    if not numeric_cols:
        print("  (no numeric columns)")
    else:
        max_vals = df[numeric_cols].max()
        max_idxs = df[numeric_cols].idxmax()
        for col in numeric_cols:
            if pd.isna(max_vals[col]):
                print(f"  {col}: all NaN")
            else:
                # +1 to convert 0-based DataFrame index to 1-based line number
                line_no = int(max_idxs[col]) + 1
                print(f"  {col}: {max_vals[col]} (line {line_no})")

    # MIN
    print("\nMinimum values per numeric column:")
    if not numeric_cols:
        print("  (no numeric columns)")
    else:
        min_vals = df[numeric_cols].min()
        min_idxs = df[numeric_cols].idxmin()
        for col in numeric_cols:
            if pd.isna(min_vals[col]):
                print(f"  {col}: all NaN")
            else:
                line_no = int(min_idxs[col]) + 1
                print(f"  {col}: {min_vals[col]} (line {line_no})")

    # NaN LOCATIONS
    print("\nNaN value locations:")
    total_nans = df.isna().sum().sum()
    if total_nans == 0:
        print("  No NaN values found.")
    else:
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                rows = df.index[df[col].isna()].tolist()
                # again +1 for 1-based
                line_numbers = [r + 1 for r in rows]
                print(f"  {col}: {nan_count} NaN(s) at lines {line_numbers}")

    # Optional: summary stats
    print("\nSummary statistics (numeric columns):")
    if numeric_cols:
        print(df[numeric_cols].describe())
    else:
        print("  (no numeric columns to summarize)")

def main():
    parser = argparse.ArgumentParser(
        description="Find max, min, and NaN locations in a CSV file."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file to analyze"
    )
    args = parser.parse_args()

    analyze_csv_stats(args.csv_file)

if __name__ == "__main__":
    main()

