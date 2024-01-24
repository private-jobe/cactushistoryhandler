import pandas as pd
import numpy as np
import argparse
from scipy.interpolate import interp1d
import sqlalchemy
import sys
from datetime import datetime

# Swedish month abbreviations to numbers
swedish_months = {
    'JAN': '01', 'FEB': '02', 'MAR': '03',
    'APR': '04', 'MAJ': '05', 'JUN': '06',
    'JUL': '07', 'AUG': '08', 'SEP': '09',
    'OKT': '10', 'NOV': '11', 'DEC': '12'
}

def swedish_date_parser(date_str):
    # Split date string into components
    day, month_abbr, year_time = date_str.split('-')
    year, time = year_time.split(' ')
    
    # Replace Swedish month abbreviation with number
    month = swedish_months.get(month_abbr.upper(), '00')

    # Reassemble date string in a format that can be parsed by to_datetime
    parseable_date_str = f"{day}-{month}-{year} {time}"
    return pd.to_datetime(parseable_date_str, format='%d-%m-%y %H:%M:%S')

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

# Parse arguments
args = parser.parse_args()

# Read data from file or database
if args.file:
    if args.interval == 0:
        print("Interval is 0. No processing has been done to the source file.")
        sys.exit(0)
    # Skip the first line (header=0) and set custom column names
    df = pd.read_csv(args.file, sep='\t', header=0, names=['timestamp', 'value', 'info'], parse_dates=['timestamp'], date_parser=swedish_date_parser)
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
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the timestamp as the index
df.set_index('timestamp', inplace=True)

# Determine start and stop times for interpolation
start_time = pd.to_datetime(args.start) if args.start else df.index.min()
stop_time = pd.to_datetime(args.stop) if args.stop else df.index.max()

# Create a new DataFrame with specified intervals
freq = f'{args.interval}S'
new_index = pd.date_range(start=start_time, end=stop_time, freq=freq)
new_df = pd.DataFrame(index=new_index)

# Interpolate data
if args.method == 'interp':
    new_df['interpolated_data'] = np.interp(new_index.astype(np.int64), 
                                            df.index.astype(np.int64), 
                                            df['value'])
elif args.method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
    interp_func = interp1d(df.index.astype(np.int64), df['value'], kind=args.method, fill_value="extrapolate")
    new_df['interpolated_data'] = interp_func(new_index.astype(np.int64))
else:
    raise ValueError(f"Unknown interpolation method: {args.method}")

# Optional: Save the interpolated data to a new CSV
new_df.to_csv('interpolated_data.csv')
