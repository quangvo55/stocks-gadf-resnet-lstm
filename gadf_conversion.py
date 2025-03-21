import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import os

# Function to load and preprocess the data
def load_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Datetime column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Sort by datetime to ensure chronological order
    df = df.sort_values('Datetime')
    
    return df

# Function to create windows of time series data by day
def create_windows(df, window_size=24, target_column='Close', step_size=1):
    """
    Create windows of time series data with specified window size, grouped by day.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data
    window_size : int
        Size of each window in hours (used for naming only)
    target_column : str
        Column to use for time series (e.g., 'Close', 'Open')
    step_size : int
        Number of steps to move forward when creating each window (not used when grouping by day)
        
    Returns:
    --------
    list of numpy arrays
        List containing windows of time series data, one per day
    list of datetime
        Corresponding timestamps for each window (last timestamp of each day)
    """
    # Ensure Datetime is datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Add a day column for grouping
    df['day'] = df['Datetime'].dt.date
    
    # Group by day and create a window for each day
    windows = []
    window_timestamps = []
    
    for day, group in df.groupby('day'):
        # Only use complete days if they have sufficient data points
        series = group[target_column].values
        
        if len(series) > 0:
            windows.append(series)
            window_timestamps.append(group['Datetime'].iloc[-1])  # Use the last timestamp of the day
    
    print(f"Created {len(windows)} daily windows")
    
    return windows, window_timestamps

# Function to convert time series windows to GADF images
def convert_to_gadf(windows):
    """
    Convert time series windows to Gramian Angular Difference Field images.
    
    Parameters:
    -----------
    windows : list of numpy arrays
        List containing windows of time series data
        
    Returns:
    --------
    list of numpy arrays
        List containing GADF images
    """
    # Initialize the GramianAngularField transformer
    gadf = GramianAngularField(method='difference')
    
    gadf_images = []
    
    for window in windows:
        # Reshape the window for the transformer
        window_reshaped = window.reshape(1, -1)
        
        # Apply min-max scaling to [-1, 1] (required for GAF)
        min_val, max_val = np.min(window), np.max(window)
        scaled_window = 2 * (window_reshaped - min_val) / (max_val - min_val) - 1
        
        # Transform to GADF
        gadf_image = gadf.fit_transform(scaled_window)
        
        # Squeeze to remove the first dimension
        gadf_images.append(gadf_image[0])
    
    return gadf_images

# Function to visualize GADF images
def visualize_gadf(gadf_images, timestamps, output_dir='gadf_images', num_samples=5):
    """
    Visualize and save GADF images.
    
    Parameters:
    -----------
    gadf_images : list of numpy arrays
        List containing GADF images
    timestamps : list of datetime
        Corresponding timestamps for each window
    output_dir : str
        Directory to save the images
    num_samples : int
        Number of sample images to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a subset of images to visualize
    if num_samples > len(gadf_images):
        num_samples = len(gadf_images)
    
    indices = np.linspace(0, len(gadf_images) - 1, num_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        timestamp = pd.to_datetime(timestamps[idx])
        timestamp_str = timestamp.strftime('%Y-%m-%d_%H')
        
        plt.figure(figsize=(8, 8))
        plt.imshow(gadf_images[idx], cmap='viridis', origin='lower')
        plt.colorbar(label='GADF Value')
        plt.title(f'GADF Image for window ending at {timestamp_str}')
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'gadf_{timestamp_str}.png'))
        plt.close()
    
    print(f"Saved {num_samples} sample GADF images to {output_dir}")

# Function to save GADF images and their corresponding future values for model training
def save_gadf_data(gadf_images, df, window_timestamps, window_size=24, 
                   prediction_horizon=1, target_column='Close', output_dir='gadf_data'):
    """
    Save GADF images and their corresponding future values for model training.
    
    Parameters:
    -----------
    gadf_images : list of numpy arrays
        List containing GADF images
    df : pandas.DataFrame
        Original DataFrame containing the time series data
    window_timestamps : list of datetime
        Timestamps corresponding to the end of each window (end of day)
    window_size : int
        Size of the window used to create the GADF images (not used)
    prediction_horizon : int
        How many days ahead to predict
    target_column : str
        Column to use for prediction targets
    output_dir : str
        Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have datetime in df
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['day'] = df['Datetime'].dt.date
    
    # Create a dictionary of days to closing prices
    day_to_close = df.groupby('day')[target_column].last().to_dict()
    
    # Prepare data for saving
    X = []  # GADF images
    y = []  # Future values
    
    for i, ts in enumerate(window_timestamps):
        current_day = pd.Timestamp(ts).date()
        
        # Calculate the future day
        future_day = pd.Timestamp(current_day) + pd.Timedelta(days=prediction_horizon)
        future_day = future_day.date()
        
        # Check if we have data for the future day
        if future_day in day_to_close:
            future_val = day_to_close[future_day]
            
            X.append(gadf_images[i])
            y.append(future_val)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Save the data
    np.save(os.path.join(output_dir, 'X_gadf.npy'), X)
    np.save(os.path.join(output_dir, 'y_targets.npy'), y)
    
    print(f"Saved {len(X)} GADF images and target values to {output_dir}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

# Main function to process the data
def process_stock_data(file_path, window_size=24, step_size=1, 
                       prediction_horizon=1, target_column='Close'):
    """
    Main function to process stock price data into GADF images.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing stock price data
    window_size : int
        Size of each window in hours (used for naming only when grouping by day)
    step_size : int
        Number of steps to move forward when creating each window (not used when grouping by day)
    prediction_horizon : int
        How many days ahead to predict
    target_column : str
        Column to use for time series and prediction targets
    """
    # Load the data
    df = load_data(file_path)
    print(f"Loaded data with {len(df)} rows")
    
    # Create windows (one per day)
    windows, window_timestamps = create_windows(
        df, window_size=window_size, target_column=target_column, step_size=step_size
    )
    
    # Convert to GADF
    gadf_images = convert_to_gadf(windows)
    print(f"Created {len(gadf_images)} GADF images (one per day)")
    
    # Visualize some sample images
    visualize_gadf(gadf_images, window_timestamps, num_samples=5)
    
    # Save data for model training
    X, y = save_gadf_data(
        gadf_images, df, window_timestamps, 
        window_size=window_size, prediction_horizon=prediction_horizon,
        target_column=target_column
    )
    
    return X, y, df, windows, gadf_images