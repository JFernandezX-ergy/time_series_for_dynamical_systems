import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import logging

# Set up logging
logging.basicConfig(filename='feature_engineering.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for data and saving figures
cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, 'data', 'interim', 'scaled_data.csv')
SAVE_DIR = os.path.join(cwd, 'reports', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# Read the data
def read_data(file_path):
    """
    Read the CSV data from the specified path.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    logging.info(f"Reading data from {file_path}")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logging.info(f"Data shape: {data.shape}")
    return data

# Feature engineering
def create_time_features(data):
    """
    Create new time-related features: hour, month, day of the week.
    
    Args:
        data (pd.DataFrame): The input dataset.
    
    Returns:
        pd.DataFrame: The dataset with new time features.
    """
    logging.info("Creating time features (hour, month, day of the week)")
    data['hour'] = data.index.hour
    data['month'] = data.index.month
    data['dayofWeek'] = data.index.dayofweek
    logging.info("Time features created successfully")
    return data

# Create rolling statistics
def add_rolling_statistics(load_data, window_width=4):
    """
    Add rolling statistics features (min, median, max) to the dataset.
    
    Args:
        load_data (pd.DataFrame): The input load column data.
        window_width (int): The window size for rolling calculations.
    
    Returns:
        pd.DataFrame: New dataframe with rolling statistics.
    """
    logging.info("Adding rolling statistics features")
    window = load_data.rolling(window=window_width)
    new_dataframe = pd.concat([window.min(), window.median(), window.max(), load_data], axis=1)
    new_dataframe.columns = ['min', 'median', 'max', 'load']
    logging.info("Rolling statistics added successfully")
    return new_dataframe

# Plot lag and autocorrelation figures
def plot_lag_autocorrelation(data, save_dir):
    """
    Plot lag plot and autocorrelation plot of the 'load' column.
    
    Args:
        data (pd.DataFrame): The input dataset containing 'load'.
        save_dir (str): Directory to save the plots.
    """
    logging.info("Plotting lag plot")
    plt.figure()
    lag_plot(data['load'])
    plt.savefig(os.path.join(save_dir, 'lag_plot.png'))
    plt.show()
    logging.info("Lag plot saved to figures directory")

    logging.info("Plotting autocorrelation plot")
    plt.figure()
    autocorrelation_plot(data['load'])
    plt.savefig(os.path.join(save_dir, 'autocorrelation_plot.png'))
    plt.show()
    logging.info("Autocorrelation plot saved to figures directory")

# Plot autocorrelation and partial autocorrelation
def plot_acf_pacf(data, save_dir):
    """
    Plot autocorrelation and partial autocorrelation of 'load'.
    
    Args:
        data (pd.DataFrame): The input dataset containing 'load'.
        save_dir (str): Directory to save the plots.
    """
    logging.info("Plotting ACF and PACF")
    plt.figure()
    plot_acf(data['load'], lags=25)
    plt.savefig(os.path.join(save_dir, 'acf_plot.png'))
    plt.show()
    
    plt.figure()
    plot_pacf(data['load'], lags=25)
    plt.savefig(os.path.join(save_dir, 'pacf_plot.png'))
    plt.show()
    logging.info("ACF and PACF plots saved to figures directory")

# Main execution
def main():
    # Step 1: Read the data
    logging.info("Starting data preprocessing")
    data = read_data(DATA_PATH)

    # Step 2: Feature engineering - Create time-related features
    data = create_time_features(data)
    logging.info("First five rows of the data after feature engineering:\n" + str(data.head()))

    # Step 3: Add rolling statistics to the 'load' column
    load_data = data[['load']]
    new_dataframe = add_rolling_statistics(load_data)
    logging.info("First ten rows of the data after adding rolling statistics:\n" + str(new_dataframe.head(10)))

    # Step 4: Drop the time-related features from the original dataframe if not needed
    data = data.drop(['hour', 'month', 'dayofWeek'], axis=1)
    logging.info("Dropped time-related features")

    # Step 5: Plot lag and autocorrelation for the load column
    plot_lag_autocorrelation(data, SAVE_DIR)

    # Step 6: Plot autocorrelation and partial autocorrelation
    plot_acf_pacf(data, SAVE_DIR)

    logging.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()
