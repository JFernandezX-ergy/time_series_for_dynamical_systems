import os
import pandas as pd


def get_data_paths():
    """
    Gets the paths for data directories.

    Returns:
        tuple: paths for raw data and interim data directories.
    """
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    raw_data_path = os.path.join(data_path, 'raw')
    interim_data_path = os.path.join(data_path, 'interim')
    return raw_data_path, interim_data_path


def read_data(raw_data_path):
    """
    Reads CSV data from the raw data path.

    Args:
        raw_data_path (str): Path to the raw data directory.

    Returns:
        pd.DataFrame: Loaded data from the CSV file.
    """
    data_file_path = os.path.join(raw_data_path, "energy.csv")
    print(f"Reading data from: {data_file_path}")
    data = pd.read_csv(data_file_path)
    return data


def preprocess_data(data):
    """
    Preprocesses the data by handling missing values and merging date and hour columns.

    Args:
        data (pd.DataFrame): Raw data to be preprocessed.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Drop NaN values
    data = data.dropna()

    # Set hour 24 to 0 (midnight)
    data.loc[data['Hour'] == 24, 'Hour'] = 0

    # Convert hour to string
    data['Hour'] = data['Hour'].astype(str)

    # Merge 'Date' and 'Hour' columns
    data['Date'] = data['Date'] + ' ' + data['Hour']
    data = data.drop(columns=['Hour'])

    # Convert 'Date' to datetime format and set as index
    datetime_series = pd.to_datetime(data['Date'], format='%d/%m/%Y %H')
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    data = data.set_index(datetime_index).drop(columns=['Date'])

    return data


def save_preprocessed_data(data, interim_data_path):
    """
    Saves the preprocessed data to the interim directory as a CSV.

    Args:
        data (pd.DataFrame): Preprocessed data.
        interim_data_path (str): Path to save the preprocessed data.
    """
    output_file_path = os.path.join(interim_data_path, 'preprocessed.csv')
    data.to_csv(output_file_path)
    print(f"Preprocessed data saved to: {output_file_path}")


if __name__ == "__main__":
    # Get paths
    raw_data_path, interim_data_path = get_data_paths()

    # Read raw data
    raw_data = read_data(raw_data_path)

    # Preprocess the data
    preprocessed_data = preprocess_data(raw_data)

    # Save the preprocessed data
    save_preprocessed_data(preprocessed_data, interim_data_path)
