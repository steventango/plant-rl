import pandas as pd

# Read the existing CSV
df = pd.read_csv('/data/online/E6/P5.1/Spreadsheet-Poisson1/z1/core.csv')

# Extract image_name from what was originally the second row (now index 1)
image_name_value = df.loc[1, 'image_name']

# Drop the first two rows
df = df.drop(index=[0, 1]).reset_index(drop=True)

# Create a new row: all NA except image_name
new_row = {col: pd.NA for col in df.columns}
new_row['image_name'] = image_name_value

# Prepend the new row
df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

# Save back to CSV
df.to_csv('/data/online/E6/P5.1/Spreadsheet-Poisson1/z1/core.csv', index=False)
