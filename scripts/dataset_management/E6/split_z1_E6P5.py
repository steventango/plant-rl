import pandas as pd

# Function to split and save CSV files
def split_and_save_csv(input_path, output_path1, output_path2, split_index):
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Split the dataframe
    df1 = df.iloc[:split_index]
    df2 = df.iloc[split_index:]
    
    # Save the split dataframes
    df1.to_csv(output_path1, index=False)
    df2.to_csv(output_path2, index=False)

# Paths for core.csv
core_input = '/data/online/E6/P5/Spreadsheet-Poisson1/z1/core.csv'
core_output1 = '/data/online/E6/P5/Spreadsheet-Poisson1/z1/core.csv'
core_output2 = '/data/online/E6/P5.1/Spreadsheet-Poisson1/z1/core.csv'

# Paths for raw.csv
raw_input = '/data/online/E6/P5/Spreadsheet-Poisson1/z1/raw.csv'
raw_output1 = '/data/online/E6/P5/Spreadsheet-Poisson1/z1/raw.csv'
raw_output2 = '/data/online/E6/P5.1/Spreadsheet-Poisson1/z1/raw.csv'

# Split both files
split_and_save_csv(core_input, core_output1, core_output2, 370)
split_and_save_csv(raw_input, raw_output1, raw_output2, 370)