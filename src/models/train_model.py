import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Turn off warnings for cleaner output
warnings.simplefilter('ignore', ConvergenceWarning)

# Paths
cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, 'data', 'interim', 'preprocessed_for_model.csv')
SAVE_DIR = os.path.join(cwd, 'reports', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# Read data
def read_data(file_path):
    """
    Read the CSV data from the specified path.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return data

# Train-Test split
def split_data(data, train_start_dt, test_start_dt, test_end_dt):
    """
    Split data into training and test sets based on given date ranges.
    
    Args:
        data (pd.DataFrame): Input dataset with DateTime index.
        train_start_dt (str): Start date for training data.
        test_start_dt (str): Start date for test data.
        test_end_dt (str): End date for test data.
    
    Returns:
        train, test (pd.DataFrame): Training and test sets.
    """
    train = data[(data.index >= train_start_dt) & (data.index < test_start_dt)][['load']]
    test = data[(data.index >= test_start_dt) & (data.index < test_end_dt)][['load']]
    return train, test

# Rescale data
def rescale_data(train, test):
    """
    Rescale training and test data between 0 and 1.
    
    Args:
        train, test (pd.DataFrame): Training and test datasets.
        
    Returns:
        train_scaled, test_scaled, scaler: Scaled data and the scaler object.
    """
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    test['load'] = scaler.transform(test)
    return train, test, scaler

# Train SARIMAX model and make predictions
def train_sarimax(train, test, horizon, order, seasonal_order, training_window=450):
    """
    Train the SARIMAX model and make predictions on the test set.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        horizon (int): Prediction horizon.
        order (tuple): ARIMA order (p, d, q).
        seasonal_order (tuple): Seasonal ARIMA order (P, D, Q, s).
        training_window (int): Size of the training window for history.
        
    Returns:
        predictions, history (list): Predicted values and training history.
    """
    test_shifted = test.copy()
    
    for t in range(1, horizon):
        test_shifted[f'load+{t}'] = test_shifted['load'].shift(-t, freq='H')
    test_shifted = test_shifted.dropna()
    
    train_ts = train['load']
    test_ts = test_shifted
    
    history = [x for x in train_ts]
    history = history[-training_window:]
    
    predictions = []
    for t in range(0, test_ts.shape[0], horizon):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        y_p = model_fit.forecast(steps=horizon)
        obs = list(test_ts.iloc[t])
        
        for j in range(horizon):
            predictions.append(y_p[j])
            history.append(y_p[j])
        for j in range(horizon):
            history.pop(0)
        print(f"Time: {test_ts.index[t]}, predicted = {y_p}, true value = {obs}")
    
    return predictions

# Plot results
def plot_results(eval_df, horizon, save_dir):
    """
    Plot and save evaluation results.
    
    Args:
        eval_df (pd.DataFrame): Evaluation dataframe with actual and predicted values.
        horizon (int): Prediction horizon.
        save_dir (str): Directory to save plots.
    """
    plt.figure()
    plt.plot(eval_df['actual'], 'k.-')
    plt.plot(eval_df['prediction'], 'x', alpha=0.7)
    plt.legend(['Actual', f'Predicted with {horizon} hr horizon'])
    plt.ylabel('Load')
    plt.xlabel('Time Index')
    plt.title('SARIMAX Predictions')
    plt.savefig(os.path.join(save_dir, 'sarimax_predictions.png'))
    plt.show()

    plt.figure()
    plt.scatter(eval_df['actual'], eval_df['prediction'], marker='*', alpha=0.8)
    plt.xlabel('True Load Values')
    plt.ylabel('Predicted Load Values')
    plt.title('True vs Predicted Load')
    lims = [2000, 4500]
    plt.xlim(lims), plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig(os.path.join(save_dir, 'sarimax_scatter_plot.png'))
    plt.show()

# Evaluate the model performance
def evaluate_model(predictions, test, scaler):
    """
    Evaluate the model's performance by comparing predicted and actual values.
    
    Args:
        predictions (list): List of predicted values.
        test (pd.DataFrame): Actual test values.
        scaler (MinMaxScaler): Scaler used for inverse transformation.
    
    Returns:
        eval_df (pd.DataFrame): Dataframe with actual and predicted values.
    """
    eval_df = pd.DataFrame(predictions, columns=['prediction'])
    eval_df = eval_df.set_index(test.index)
    eval_df['actual'] = test['load']
    
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    
    return eval_df

# Main function
def main():
    data = read_data(DATA_PATH)
    
    # Train-Test Split
    train, test = split_data(data, '2012-09-01', '2014-11-01', '2014-11-04')
    
    # Rescale Data
    train_scaled, test_scaled, scaler = rescale_data(train, test)
    
    # Hyperparameters
    horizon = 6
    order = (3, 1, 6)
    seasonal_order = (1, 1, 1, 24)
    
    # Train SARIMAX model and make predictions
    predictions = train_sarimax(train_scaled, test_scaled, horizon, order, seasonal_order)
    
    # Evaluate model performance
    eval_df = evaluate_model(predictions, test, scaler)
    
    # Plot results
    plot_results(eval_df, horizon, SAVE_DIR)
    
    # R^2 score
    score = r2_score(eval_df['actual'], eval_df['prediction'])
    print(f'R^2 score: {score:.2f}')

if __name__ == "__main__":
    main()
