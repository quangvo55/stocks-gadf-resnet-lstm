import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Import from our GADF conversion script
from gadf_conversion import process_stock_data, convert_to_gadf, create_windows

# Import from our ResNet model script
from resnet_prediction_model import train_stock_prediction_model, predict_stock_prices, resnet18, resnet34

def run_complete_pipeline(csv_file_path, window_size=24, step_size=1, prediction_horizon=1,
                         target_column='Close', model_type='resnet18_gru', batch_size=32, 
                         num_epochs=50, test_size=0.2, val_size=0.2,
                         rnn_hidden_size=128, rnn_num_layers=1, dropout=0.2):
    """
    Run the complete pipeline from CSV data to trained prediction model.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to CSV file with stock data
    window_size : int
        Size of each window in hours (now just used for naming)
    step_size : int
        Number of steps to move forward when creating each window (not used with daily grouping)
    prediction_horizon : int
        How many days ahead to predict
    target_column : str
        Column to use for time series and prediction targets
    model_type : str
        Type of model to use:
        - 'resnet18_gru': ResNet-18 with GRU
        - 'resnet18_lstm': ResNet-18 with LSTM
        - 'resnet34_gru': ResNet-34 with GRU
        - 'resnet34_lstm': ResNet-34 with LSTM
        - 'resnet18': Legacy option (uses GRU)
        - 'resnet34': Legacy option (uses GRU)
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of epochs to train for
    test_size : float
        Proportion of data to use for testing
    val_size : float
        Proportion of training data to use for validation
    rnn_hidden_size : int
        Hidden size for the RNN layer
    rnn_num_layers : int
        Number of RNN layers
    dropout : float
        Dropout rate for regularization
        
    Returns:
    --------
    model : PyTorch model
        Trained ResNet-RNN hybrid model
    results : dict
        Evaluation results
    df : pandas.DataFrame
        Original stock data
    """
    # Create timestamp for output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"predictions/prediction_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process the stock data to create GADF images
    print("Step 1: Converting time series to GADF images...")
    X, y, df, windows, gadf_images = process_stock_data(
        csv_file_path, window_size=window_size, step_size=step_size,
        prediction_horizon=prediction_horizon, target_column=target_column
    )
    
    # Save some parameters for future reference
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        f.write(f"CSV file: {csv_file_path}\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Prediction horizon: {prediction_horizon}\n")
        f.write(f"Target column: {target_column}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"RNN hidden size: {rnn_hidden_size}\n")
        f.write(f"RNN number of layers: {rnn_num_layers}\n")
        f.write(f"Dropout rate: {dropout}\n")
        f.write(f"Number of GADF images: {len(gadf_images)}\n")
        f.write(f"Data shape - X: {X.shape}, y: {y.shape}\n")
    
    # Step 2: Train the ResNet-RNN hybrid model
    print(f"\nStep 2: Training {model_type} model...")
    model, results = train_stock_prediction_model(
        X, y, 
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=num_epochs,
        output_dir=os.path.join(output_dir, 'model_output'),
        test_size=test_size,
        val_size=val_size,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        dropout=dropout
    )
    
    # Step 3: Visualize some example predictions
    print("\nStep 3: Visualizing example predictions...")
    visualize_example_predictions(model, X, y, results, 
                                  output_dir=os.path.join(output_dir, 'visualizations'))
    
    print(f"\nPipeline completed. All outputs saved to {output_dir}")
    return model, results, df

def visualize_example_predictions(model, X, y, results, output_dir, num_examples=5):
    """
    Visualize some example predictions alongside the original GADF images.
    
    Parameters:
    -----------
    model : PyTorch model
        Trained ResNet model
    X : numpy array
        GADF images
    y : numpy array
        Target values
    results : dict
        Evaluation results from the model
    output_dir : str
        Directory to save visualizations
    num_examples : int
        Number of examples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get indices of test samples
    test_indices = list(range(len(results['predictions'])))
    
    # Select a few random examples
    if num_examples > len(test_indices):
        num_examples = len(test_indices)
    
    # Use evenly spaced examples
    example_indices = np.linspace(0, len(test_indices) - 1, num_examples, dtype=int)
    selected_indices = [test_indices[i] for i in example_indices]
    
    # Visualize each example
    for i, idx in enumerate(selected_indices):
        plt.figure(figsize=(15, 5))
        
        # Plot the GADF image
        plt.subplot(1, 3, 1)
        plt.imshow(X[idx], cmap='viridis', origin='lower')
        plt.colorbar(label='GADF Value')
        plt.title('GADF Image')
        
        # Plot the actual vs predicted value
        actual = results['targets'][i]
        predicted = results['predictions'][i]
        
        plt.subplot(1, 3, 2)
        plt.bar(['Actual', 'Predicted'], [actual, predicted])
        plt.title(f'Stock Price: Actual vs Predicted')
        plt.ylabel('Price')
        
        # Plot the prediction error
        plt.subplot(1, 3, 3)
        error = predicted - actual
        error_percent = (error / actual) * 100
        plt.bar(['Absolute Error', 'Percent Error (%)'], [error, error_percent])
        plt.title('Prediction Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'example_{i+1}.png'))
        plt.close()
    
    print(f"Saved {num_examples} example visualizations to {output_dir}")

# Function to make predictions for future stock prices
def predict_future_prices(model, df, prediction_steps=5, target_column='Close'):
    """
    Predict future stock prices based on the most recent data.
    
    Parameters:
    -----------
    model : PyTorch model
        Trained ResNet model
    df : pandas.DataFrame
        Original stock data
    prediction_steps : int
        Number of days ahead to predict
    target_column : str
        Column to use for prediction
        
    Returns:
    --------
    list
        Predicted future prices
    """
    # Ensure we have datetime in df
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['day'] = df['Datetime'].dt.date
    
    # Get the most recent day's data
    latest_day = df['day'].max()
    latest_day_data = df[df['day'] == latest_day][target_column].values
    
    # Create a window for the latest day
    windows = [latest_day_data]
    
    # Convert to GADF
    gadf_images = convert_to_gadf(windows)
    
    # Make initial prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Predict next day's closing price
    gadf_image = gadf_images[0]
    gadf_image = np.expand_dims(gadf_image, axis=0)  # Add batch dimension
    next_day_pred = predict_stock_prices(model, gadf_image, device=device)[0]
    
    # If we only want one day ahead
    if prediction_steps == 1:
        return [next_day_pred]
    
    # For multi-day predictions, we need to simulate future days
    # We'll create synthetic daily data based on the predicted closing price
    predictions = [next_day_pred]
    
    # For subsequent days, create synthetic daily patterns based on the last day,
    # but adjusted for the new predicted closing price
    for day in range(1, prediction_steps):
        # Use the last day's pattern as a template, but scale to the new closing price
        previous_day_data = latest_day_data
        previous_close = previous_day_data[-1]
        next_close = predictions[-1]
        
        # Scale factor based on predicted change
        scale_factor = next_close / previous_close
        
        # Create synthetic day data by scaling the previous day's pattern
        synthetic_day_data = previous_day_data * scale_factor
        
        # Convert to GADF
        synthetic_gadf = convert_to_gadf([synthetic_day_data])[0]
        synthetic_gadf = np.expand_dims(synthetic_gadf, axis=0)
        
        # Predict the next day
        next_day_pred = predict_stock_prices(model, synthetic_gadf, device=device)[0]
        predictions.append(next_day_pred)
    
    return predictions

def plot_future_predictions(df, future_predictions, target_column='Close', n_past_days=30):
    """
    Plot historical data and future predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original stock data
    future_predictions : list
        Predicted future prices
    target_column : str
        Column to use for prediction
    n_past_days : int
        Number of past days to show
    """
    # Ensure we have datetime in df
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['day'] = df['Datetime'].dt.date
    
    # Get daily data for the past n days
    daily_data = df.groupby('day')[target_column].last()
    recent_data = daily_data.values[-n_past_days:]
    recent_dates = daily_data.index[-n_past_days:]
    
    # Create future dates (assuming daily data)
    last_date = daily_data.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(len(future_predictions))]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(recent_dates, recent_data, 'b-', label='Historical')
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, 'r--', label='Predicted')
    
    # Add a vertical line to separate historical from predictions
    plt.axvline(x=last_date, color='green', linestyle='--', alpha=0.7)
    
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis date ticks
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    csv_file_path = "data/LUNR_15m_ohlc.csv"
    
    # Run the complete pipeline with ResNet-GRU hybrid model
    model, results, df = run_complete_pipeline(
        csv_file_path,
        window_size=24,  # This is now just used for naming
        step_size=1,     # Not used with daily grouping
        prediction_horizon=1,  # Predict 1 day ahead
        target_column='Close',
        model_type='resnet18_gru',  # Using GRU by default
        batch_size=32,
        num_epochs=30,  # Reduced for demonstration
        rnn_hidden_size=128,
        rnn_num_layers=1,
        dropout=0.2
    )
    
    # Predict future prices (next 5 trading days)
    future_predictions = predict_future_prices(
        model, df, prediction_steps=5, target_column='Close'
    )
    
    # Plot future predictions
    plot_future_predictions(df, future_predictions, target_column='Close', n_past_days=30)
    
    print("Future predictions (next trading days):")
    for i, pred in enumerate(future_predictions):
        print(f"Day t+{i+1}: {pred:.2f}")