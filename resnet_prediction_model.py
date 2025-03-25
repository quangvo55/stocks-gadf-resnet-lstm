import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
from datetime import datetime

# Custom dataset for GADF images and targets
class GADFDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        target = self.y[idx]
        
        # Add channel dimension for PyTorch (C, H, W)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.float).view(1)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, target_tensor

# Define the ResNet block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# Define the ResNet-RNN hybrid model
class ResNetRNN(nn.Module):
    def __init__(self, block, layers, rnn_type='gru', hidden_size=128, num_rnn_layers=1, dropout=0.2, num_classes=1):
        super(ResNetRNN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension after CNN
        self.feature_dim = 512
        
        # Choose between LSTM and GRU
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_size,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=dropout if num_rnn_layers > 1 else 0
            )
        else:  # Default to GRU
            self.rnn = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=hidden_size,
                num_layers=num_rnn_layers,
                batch_first=True,
                dropout=dropout if num_rnn_layers > 1 else 0
            )
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]
        
        # Reshape for RNN input: [batch_size, sequence_length, features]
        # For a single image, sequence_length = 1
        x = x.unsqueeze(1)
        
        # RNN processing
        if self.rnn_type == 'lstm':
            x, (_, _) = self.rnn(x)
        else:  # GRU
            x, _ = self.rnn(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Final prediction
        x = self.fc(x)
        
        return x

# Create ResNet-18-RNN hybrid models
def resnet18_lstm(hidden_size=128, num_rnn_layers=1, dropout=0.2):
    return ResNetRNN(ResidualBlock, [2, 2, 2, 2], rnn_type='lstm', 
                  hidden_size=hidden_size, num_rnn_layers=num_rnn_layers, dropout=dropout)

def resnet18_gru(hidden_size=128, num_rnn_layers=1, dropout=0.2):
    return ResNetRNN(ResidualBlock, [2, 2, 2, 2], rnn_type='gru', 
                  hidden_size=hidden_size, num_rnn_layers=num_rnn_layers, dropout=dropout)

# Create ResNet-34-RNN hybrid models
def resnet34_lstm(hidden_size=128, num_rnn_layers=1, dropout=0.2):
    return ResNetRNN(ResidualBlock, [3, 4, 6, 3], rnn_type='lstm', 
                  hidden_size=hidden_size, num_rnn_layers=num_rnn_layers, dropout=dropout)

def resnet34_gru(hidden_size=128, num_rnn_layers=1, dropout=0.2):
    return ResNetRNN(ResidualBlock, [3, 4, 6, 3], rnn_type='gru', 
                  hidden_size=hidden_size, num_rnn_layers=num_rnn_layers, dropout=dropout)

# Legacy functions for backward compatibility
def resnet18():
    return resnet18_gru()  # Default to GRU

def resnet34():
    return resnet34_gru()  # Default to GRU

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    model.to(device)
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        # Learning rate scheduler step
        scheduler.step(epoch_val_loss)
        
        # Save the best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = model.state_dict().copy()
            
        print()
        
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Evaluation function
def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and targets
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    test_loss = running_loss / len(test_loader.dataset)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    results = {
        'test_loss': test_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results

# Plot training history
def plot_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

# Plot predictions vs actual values
def plot_predictions(results, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Plot predictions vs targets
    plt.subplot(1, 2, 1)
    plt.scatter(results['targets'], results['predictions'], alpha=0.5)
    
    # Add the perfect prediction line
    min_val = min(min(results['targets']), min(results['predictions']))
    max_val = max(max(results['targets']), max(results['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual')
    
    # Plot histogram of errors
    plt.subplot(1, 2, 2)
    errors = results['predictions'] - results['targets']
    plt.hist(errors, bins=20)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    # Print metrics
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"R²: {results['r2']:.4f}")

# Main function to train the model
def train_stock_prediction_model(X, y, model_type='resnet18_gru', batch_size=32, num_epochs=50, 
                                output_dir='model_output', test_size=0.2, val_size=0.2,
                                rnn_hidden_size=128, rnn_num_layers=1, dropout=0.2):
    """
    Train a ResNet-RNN hybrid model to predict stock prices from GADF images.
    
    Parameters:
    -----------
    X : numpy array
        GADF images
    y : numpy array
        Target values (future stock prices)
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
    output_dir : str
        Directory to save model outputs
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
        Trained ResNet-RNN model
    results : dict
        Evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train/test split (chronological)
    train_size = int((1 - test_size) * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train/validation split (chronological)
    train_val_size = int((1 - val_size) * len(X_train))
    X_train_final, X_val = X_train[:train_val_size], X_train[train_val_size:]
    y_train_final, y_val = y_train[:train_val_size], y_train[train_val_size:]
    
    print(f"Training set: {len(X_train_final)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = GADFDataset(X_train_final, y_train_final)
    val_dataset = GADFDataset(X_val, y_val)
    test_dataset = GADFDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model based on type
    if model_type == 'resnet18_gru':
        model = resnet18_gru(hidden_size=rnn_hidden_size, num_rnn_layers=rnn_num_layers, dropout=dropout)
    elif model_type == 'resnet18_lstm':
        model = resnet18_lstm(hidden_size=rnn_hidden_size, num_rnn_layers=rnn_num_layers, dropout=dropout)
    elif model_type == 'resnet34_gru':
        model = resnet34_gru(hidden_size=rnn_hidden_size, num_rnn_layers=rnn_num_layers, dropout=dropout)
    elif model_type == 'resnet34_lstm':
        model = resnet34_lstm(hidden_size=rnn_hidden_size, num_rnn_layers=rnn_num_layers, dropout=dropout)
    elif model_type == 'resnet18':
        model = resnet18()  # Legacy support
    elif model_type == 'resnet34':
        model = resnet34()  # Legacy support
    else:
        raise ValueError("Invalid model_type. Choose from 'resnet18_gru', 'resnet18_lstm', 'resnet34_gru', 'resnet34_lstm'")
    
    # Log model architecture and hyperparameters
    with open(os.path.join(output_dir, 'model_architecture.txt'), 'w') as f:
        f.write(f"Model type: {model_type}\n")
        f.write(f"RNN hidden size: {rnn_hidden_size}\n")
        f.write(f"RNN number of layers: {rnn_num_layers}\n")
        f.write(f"Dropout rate: {dropout}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: 0.001\n")
        f.write("\nModel Architecture:\n")
        f.write(str(model))
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train the model
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epochs=num_epochs, device=device
    )
    
    # Evaluate on test set
    results = evaluate_model(trained_model, test_loader, criterion, device)
    
    # Plot and save results
    plot_history(history, save_path=os.path.join(output_dir, 'training_history.png'))
    plot_predictions(results, save_path=os.path.join(output_dir, 'predictions.png'))
    
    # Save the model
    model_save_path = os.path.join(output_dir, f'stock_prediction_{model_type}.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return trained_model, results

# Function to make predictions on new data
def predict_stock_prices(model, new_gadf_images, device='cpu'):
    """
    Make predictions on new GADF images.
    
    Parameters:
    -----------
    model : PyTorch model
        Trained ResNet model
    new_gadf_images : numpy array
        New GADF images to predict on
    device : str
        Device to use for prediction
        
    Returns:
    --------
    numpy array
        Predicted stock prices
    """
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = GADFDataset(new_gadf_images, np.zeros(len(new_gadf_images)))
    dataloader = DataLoader(dataset, batch_size=32)
    
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions).flatten()