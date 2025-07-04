import numpy as np
import pandas as pd


def load_and_clean_data(spectral_file: str) -> tuple[pd.DataFrame, dict]:
    """Loads spectral data from a file and cleans it."""
    df = pd.read_csv(spectral_file, sep="\t", header=0)

    # Clean the DataFrame
    df = df.iloc[:, :-1]  # Drop last column
    df = df.drop(df.columns[1], axis=1)  # Drop second column
    df = df.drop(df.columns[range(1, df.shape[1], 2)], axis=1)  # Drop odd columns

    # If there are more than 14 columns, drop the second column
    if df.shape[1] > 14:
        df = df.drop(df.columns[1], axis=1)

    df.columns = ["Wavelength"] + df.columns[1:].tolist()

    # Calculate integrals and sort
    integrals = calculate_integrals(df)

    # sort the dataframe columns by the integral values
    sorted_action_values = sorted(integrals, key=lambda x: integrals[x])
    df = df[["Wavelength"] + sorted_action_values]

    columns = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Rename columns
    df.columns = ["Wavelength"] + [col for col in columns[: df.shape[1] - 1]]

    integrals = calculate_integrals(df)

    return df, integrals


def calculate_integrals(df):
    integrals = {}
    for col in df.columns[1:]:
        # between wavelengths of 700 nm and 750 nm
        df_filtered = df[(df["Wavelength"] >= 700) & (df["Wavelength"] <= 750)]
        integral = np.trapz(df_filtered[col], df_filtered["Wavelength"])
        if integral < 0:
            integral = 0
        integrals[col] = integral
    return integrals
