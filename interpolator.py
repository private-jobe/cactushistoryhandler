import pandas as pd
import numpy as np
import argparse
from scipy.interpolate import interp1d
import sqlalchemy
import sys
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Swedish month abbreviations to numbers
swedish_months = {
    'JAN': '01', 'FEB': '02', 'MAR': '03',
    'APR': '04', 'MAJ': '05', 'JUN': '06',
    'JUL': '07', 'AUG': '08', 'SEP': '09',
    'OKT': '10', 'NOV': '11', 'DEC': '12'
}

def cactus_date_parser(date_str):
    # Split date string into components
    day, month_abbr, year_time = date_str.split('-')
    year, time = year_time.split(' ')
    
    # Replace Swedish month abbreviation with number
    month = swedish_months.get(month_abbr.upper(), '00')

    # Reassemble date string in a format that can be parsed by to_datetime
    parseable_date_str = f"{year}-{month}-{day} {time}"
    return parseable_date_str
    #return pd.to_datetime(parseable_date_str, format="%y-%m-%d %H:%M:%S")
    #return pd.to_datetime(parseable_date_str, format='%d-%m-%y %H:%M:%S')

# Setup argument parser
parser = argparse.ArgumentParser(description='Interpolate time series data to a specified interval.')
parser.add_argument('--interval', type=int, help='Interval time in seconds between interpolations', default=0)
parser.add_argument('--start', type=str, help='Start date and time in YYYY-MM-DD HH:MM:SS format', default=None)
parser.add_argument('--stop', type=str, help='Stop date and time in YYYY-MM-DD HH:MM:SS format', default=None)
parser.add_argument('--method', type=str, help='Interpolation method', default='interp')
parser.add_argument('--file', type=str, help='Location of the CSV file to read data from', default=None)
parser.add_argument('--sep', type=str, help='Data separator', default=None)
parser.add_argument('--dbstring', type=str, help='Database connection string', default=None)
parser.add_argument('--table', type=str, help='Table name to query data from', default=None)
parser.add_argument('--plot', action='store_true', help='Plot the original and interpolated data')

# Parse arguments
args = parser.parse_args()

# Set default output filepath
output_filepath = None

# Read data from file or database
if args.file:

    # Update output_filepath
    output_filepath = args.file

    if args.interval == 0:
        print("Interval is 0. No processing has been done to the source file.")
        sys.exit(0)

    # Read the first line to get the original header
    with open(args.file, 'r') as f:
        original_header = f.readline().strip()
    
    # Skip the first line (header=0) and set custom column names
    df = pd.read_csv(args.file, sep='\t', header=0, names=['timestamp', original_header, 'info'])

    # Apply the custom date parser to the timestamp column
    df['timestamp'] = df['timestamp'].apply(cactus_date_parser)

    # Ensure timestamp is in the correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%y-%m-%d %H:%M:%S")

elif args.dbstring and args.table:
    engine = sqlalchemy.create_engine(args.dbstring)
    df = pd.read_sql_table(args.table, engine)
    if args.interval == 0:
        df.to_csv('database_data.csv')
        print("Interval is 0. Database data has been passed through without interpolation.")
        sys.exit(0)
else:
    raise ValueError("Either --file or --dbstring and --table must be provided.")

# Ensure timestamp is in the correct format
#df['timestamp'] = pd.to_datetime(df['timestamp'], format="%y-%m-%d %H:%M:%S")

# Set the timestamp as the index
df.set_index('timestamp', inplace=True)

# Handle duplicates by averaging
df = df.groupby(df.index).mean()

# Determine start and stop times for interpolation
start_time = pd.to_datetime(args.start) if args.start else df.index.min()
stop_time = pd.to_datetime(args.stop) if args.stop else df.index.max()

# Create a new DataFrame with specified intervals
freq = f'{args.interval}s'
new_index = pd.date_range(start=start_time, end=stop_time, freq=freq)
new_df = pd.DataFrame(index=new_index)
new_df.index.name = "timestamp"

# Set new column name
column_name = original_header

# Interpolate data
if args.method == 'interp':
    new_df[column_name] = np.interp(new_index.astype(np.int64), 
                                            df.index.astype(np.int64), 
                                            df[original_header])
elif args.method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
    interp_func = interp1d(df.index.astype(np.int64), df[original_header], kind=args.method, fill_value="extrapolate")
    new_df[column_name] = interp_func(new_index.astype(np.int64))
else:
    raise ValueError(f"Unknown interpolation method: {args.method}")

# truncating value
new_df[column_name] = np.trunc(10 * new_df[column_name]) / 10

# Generate new filename
if output_filepath == None:
    output_filepath = "./data/data.txt"
filepath = os.path.split(output_filepath)[0]
filename = os.path.splitext(os.path.basename(output_filepath))[0]
fileext = os.path.splitext(os.path.basename(output_filepath))[1]
new_filename = filename + "_" + args.method
new_filepath = filepath + "/" + new_filename + fileext

# Optional: Save the interpolated data to a new CSV
new_df.to_csv(new_filepath, sep='\t')

# Evaluate errors (assuming you have true values for comparison)
true_values = df[original_header]
interpolated_values = new_df[column_name].reindex(df.index, method='nearest')

mae = mean_absolute_error(true_values, interpolated_values)
mse = mean_squared_error(true_values, interpolated_values)
rmse = np.sqrt(mse)
r2 = r2_score(true_values, interpolated_values)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}")

# Check if plotting is required
if args.plot:
    # Turn on the interactive mode
    plt.ion()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[original_header], label='Original Data', marker='o')
    plt.plot(new_df.index, new_df[column_name], label='Interpolated Data', marker='x', linestyle='--')
    plt.title('Original vs Interpolated Data')
    plt.xlabel('Timestamp')
    plt.ylabel(original_header)
    plt.legend()
    plt.show()


sec = input('Press any key to exit.')
