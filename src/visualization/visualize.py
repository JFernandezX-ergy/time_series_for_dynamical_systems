import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
from sklearn.preprocessing import StandardScaler

def read_preprocessed_data():
    """
    Reads the preprocessed data from the interim folder.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    cwd = os.getcwd()  # Current working directory
    data_path = os.path.join(cwd, 'data', 'interim', 'preprocessed.csv')
    print(f"Reading data from: {data_path}")

    # Reading the CSV file
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    return data


def setup_save_dir():
    """
    Set up the directory to save figures.
    Returns:
        str: Path to the directory where figures will be saved.
    """
    save_dir_figures = os.path.join(os.getcwd(), 'reports', 'figures')
    save_dir_scaled_data = os.path.join(os.getcwd(), 'data', 'interim')
    
    os.makedirs(save_dir_figures, exist_ok=True)
    os.makedirs(save_dir_scaled_data, exist_ok=True)
    return save_dir_figures, save_dir_scaled_data

def filter_data_by_date(data, date):
    """
    Filter data by a specific date.

    Args:
        data (pd.DataFrame): The data to filter.
        date (str): The date string to filter by.
    
    Returns:
        pd.DataFrame: Filtered data for the specific date.
    """
    return data.loc[date]

def truncate_data(data, start_date, end_date):
    """
    Truncate data between two dates.

    Args:
        data (pd.DataFrame): The data to truncate.
        start_date (str): The start date for truncation.
        end_date (str): The end date for truncation.
   
    Returns:
        pd.DataFrame: Truncated data.
    """
    # Sort the index to ensure truncation works correctly
    data = data.sort_index()
    return data.truncate(before=start_date, after=end_date)


def adjust_frequency(data, freq='B', method='pad'):
    """
    Adjust the frequency of the data by resampling.

    Args:
        data (pd.DataFrame): The input time series data.
        freq (str): The desired frequency (e.g., 'B' for business days).
        method (str): Method to use for filling holes in reindexed data.

    Returns:
        pd.DataFrame: Data resampled with the new frequency.
    """
    # Ensure the index is sorted to avoid errors
    data = data.sort_index()
    return data.asfreq(freq, method=method)


def seasonal_decompose(data_load, start_date, end_date, model='additive'):
    """
    Perform seasonal decomposition on the load data between specified dates.

    Args:
        data_load (pd.Series): The time series data to decompose.
        start_date (str): The start date for the decomposition.
        end_date (str): The end date for the decomposition.
        model (str): The decomposition model (default is 'additive').

    Returns:
        DecompositionResult: The result of the seasonal decomposition.
    """
    # Ensure the index is sorted to avoid KeyError for non-monotonic indexes
    data_load = data_load.sort_index()

    # Filter the data within the provided date range
    filtered_data = data_load[start_date:end_date]

    # Perform seasonal decomposition
    return sm.tsa.seasonal_decompose(filtered_data, model=model)


def plot_and_save(fig, file_name, save_dir):
    """
    Save the plot to the specified directory.
 
    Args:
        fig: Matplotlib figure to be saved.
        file_name (str): The name to save the figure as.
        save_dir (str): Directory to save the figure.
    """
    save_path = os.path.join(save_dir, file_name)
    fig.savefig(save_path)
    plt.show()

def plot_with_time_locators(data_load, decomposition_trend, save_dir):
    """
    Plot time series data with custom time locators.

    Args:
        data_load (pd.Series): Load data to plot.
        decomposition_trend (pd.Series): Decomposed trend data.
        save_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots()
    ax.grid(True)

    year_locator = mdates.YearLocator()
    month_locator = mdates.MonthLocator(interval=1)
    year_formatter = mdates.DateFormatter('%Y')
    month_formatter = mdates.DateFormatter('%m')

    ax.xaxis.set_minor_locator(month_locator)
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(year_formatter)

    colors = mcolors.CSS4_COLORS
    ax.plot(data_load.index, data_load, color=colors['lightskyblue'])
    ax.plot(decomposition_trend.index, decomposition_trend, color=colors['midnightblue'])

    mpl.rcParams['figure.figsize'] = [18.0, 6.0]

    plot_and_save(fig, 'trend_vs_load.png', save_dir)

def rescale_data(data):
    """
    Rescale the data using StandardScaler.

    Args:
        data (pd.DataFrame): The data to be scaled.
   
    Returns:
        pd.DataFrame: Scaled data.
    """
    scaler = StandardScaler()
    features_to_scale = data.columns
    data.loc[:, features_to_scale] = scaler.fit_transform(data[features_to_scale])
    return data

def main():
    """
    Main function to execute the entire data pipeline.
    """
    data = read_preprocessed_data()

    # Assuming 'data' is already loaded somewhere
    save_dir_figures, save_dir_scaled_data = setup_save_dir()

    # Example: Filter data by a specific date
    filtered_data = filter_data_by_date(data, '2012-1-5')
    print(filtered_data)

    # Truncate data between two dates
    truncated_data = truncate_data(data, '2012-11-01', '2012-11-02')

    # Adjust frequency and fill missing values
    data_custom = adjust_frequency(data)

    # Seasonal decomposition
    data_load = data['load']
    # Call the function with the correct date range
    decomposition = seasonal_decompose(data_load, '2014-06-01', '2014-12-31')

    # Plot and save seasonal decomposition
    fig = decomposition.plot()
    plot_and_save(fig, 'seasonal_decomposition.png', save_dir_figures)

    # Plot with custom time locators
    plot_with_time_locators(data_load, decomposition.trend, save_dir_figures)

    # Rescale the data
    scaled_data = rescale_data(data)
    scaled_data.to_csv(os.path.join(save_dir_scaled_data, 'scaled_data.csv'))

if __name__ == "__main__":
    main()
