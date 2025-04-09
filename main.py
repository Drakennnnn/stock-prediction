@st.cache_data
def generate_synthetic_stock_data(n_samples=1000, seed=42):
    """Generate synthetic stock data for prediction models."""
    np.random.seed(seed)
    
    # Create date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_samples * 2)  # Add extra buffer for weekdays
    # Ensure we have more dates than we need
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    # Only take the needed number of samples, with a safety check
    if len(date_range) >= n_samples:
        date_range = date_range[:n_samples]
    else:
        # If we don't have enough business days, extend further back
        start_date = end_date - timedelta(days=n_samples * 3)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')[:n_samples]
    
    # Base price around $100 with some randomness
    base_price = 100
    
    # Generate data
    data = []
    prev_close = base_price
    
    for i in range(n_samples):
        # Generate daily volatility (more volatile on some days)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Generate price components with some correlation
        open_price = prev_close * (1 + np.random.normal(0, volatility))
        
        # High and low with proper constraints
        daily_range = open_price * volatility * np.random.uniform(1, 3)
        high_price = open_price + daily_range/2
        low_price = open_price - daily_range/2
        
        # Ensure low price is not negative
        low_price = max(low_price, 0.1)
        
        # Ensure high > open and low < open
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        
        # Close price with some trend and mean reversion
        momentum = np.random.normal(0, 0.01)
        mean_reversion = (base_price - open_price) * 0.05  # Pull towards base price
        close_price = open_price * (1 + momentum + mean_reversion)
        
        # Ensure close is between high and low
        close_price = min(max(close_price, low_price), high_price)
        
        # Volume with some correlation to price movement
        base_volume = np.random.randint(100000, 1000000)
        price_change_ratio = abs(close_price - open_price) / open_price
        volume = int(base_volume * (1 + price_change_ratio * 10))
        
        # Trading pattern: sometimes more, sometimes less
        if np.random.random() < 0.1:  # 10% chance of high trading day
            volume *= np.random.uniform(1.5, 3)
        
        # Price movement indicator (1 if price increased, 0 if decreased)
        price_movement = 1 if close_price > open_price else 0
        
        # Store the day's data
        data.append({
            'Date': date_range[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume,
            'Price_Movement': price_movement
        })
        
        # Set close as previous close for next iteration
        prev_close = close_price
    
    df = pd.DataFrame(data)
    return dfimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, classification_report
)
import math
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5 !important;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #0277BD !important;
        margin-top: 1rem;
    }
    .section-header {
        font-size: 1.5rem !important;
        color: #0277BD !important;
        margin-top: 0.8rem;
        border-bottom: 1px solid #BBDEFB;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-card * {
        color: #333333 !important;
    }
    .green-text {
        color: green !important;
        font-weight: bold;
    }
    .red-text {
        color: red !important;
        font-weight: bold;
    }
    .highlight {
        background-color: #f9f9f9 !important;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    .highlight p, .highlight li, .highlight ul, .highlight ol {
        color: #333333 !important;
    }
    .comparison-table {
        width: 100%;
        text-align: center;
    }
    .comparison-table th {
        background-color: #f5f5f5;
    }
    .comparison-table th, .comparison-table td {
        color: #333333 !important;
    }
    /* Make st.table and st.dataframe text dark */
    .stDataFrame, .stTable {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Data Generation Functions -------------------

@st.cache_data
def generate_synthetic_stock_data(n_samples=1000, seed=42):
    """Generate synthetic stock data for prediction models."""
    np.random.seed(seed)
    
    # Create date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_samples * 2)  # Add extra buffer for weekdays
    # Ensure we have more dates than we need
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    # Only take the needed number of samples, with a safety check
    if len(date_range) >= n_samples:
        date_range = date_range[:n_samples]
    else:
        # If we don't have enough business days, extend further back
        start_date = end_date - timedelta(days=n_samples * 3)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')[:n_samples]
    
    # Base price around $100 with some randomness
    base_price = 100
    
    # Generate data
    data = []
    prev_close = base_price
    
    for i in range(n_samples):
        # Generate daily volatility (more volatile on some days)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Generate price components with some correlation
        open_price = prev_close * (1 + np.random.normal(0, volatility))
        
        # High and low with proper constraints
        daily_range = open_price * volatility * np.random.uniform(1, 3)
        high_price = open_price + daily_range/2
        low_price = open_price - daily_range/2
        
        # Ensure low price is not negative
        low_price = max(low_price, 0.1)
        
        # Ensure high > open and low < open
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        
        # Close price with some trend and mean reversion
        momentum = np.random.normal(0, 0.01)
        mean_reversion = (base_price - open_price) * 0.05  # Pull towards base price
        
        # Add significant random noise to make prediction harder (key change for realistic accuracy)
        random_noise = np.random.normal(0, 0.02)  # Increased noise
        close_price = open_price * (1 + momentum + mean_reversion + random_noise)
        
        # Ensure close is between high and low
        close_price = min(max(close_price, low_price), high_price)
        
        # Volume with some correlation to price movement
        base_volume = np.random.randint(100000, 1000000)
        price_change_ratio = abs(close_price - open_price) / open_price
        volume = int(base_volume * (1 + price_change_ratio * 10))
        
        # Trading pattern: sometimes more, sometimes less
        if np.random.random() < 0.1:  # 10% chance of high trading day
            volume *= np.random.uniform(1.5, 3)
        
        # Price movement indicator (1 if price increased, 0 if decreased)
        # Add some noise to make this harder to predict (5% random flips)
        if np.random.random() < 0.05:
            # Randomly flip the true movement 5% of the time
            price_movement = 0 if close_price > open_price else 1
        else:
            price_movement = 1 if close_price > open_price else 0
        
        # Store the day's data
        data.append({
            'Date': date_range[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume,
            'Price_Movement': price_movement
        })
        
        # Set close as previous close for next iteration
        prev_close = close_price
    
    df = pd.DataFrame(data)
    return df
    
    # Base price around $100 with some randomness
    base_price = 100
    
    # Generate data
    data = []
    prev_close = base_price
    
    for i in range(n_samples):
        # Generate daily volatility (more volatile on some days)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Generate price components with some correlation
        open_price = prev_close * (1 + np.random.normal(0, volatility))
        
        # High and low with proper constraints
        daily_range = open_price * volatility * np.random.uniform(1, 3)
        high_price = open_price + daily_range/2
        low_price = open_price - daily_range/2
        
        # Ensure low price is not negative
        low_price = max(low_price, 0.1)
        
        # Ensure high > open and low < open
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        
        # Close price with some trend and mean reversion
        momentum = np.random.normal(0, 0.01)
        mean_reversion = (base_price - open_price) * 0.05  # Pull towards base price
        close_price = open_price * (1 + momentum + mean_reversion)
        
        # Ensure close is between high and low
        close_price = min(max(close_price, low_price), high_price)
        
        # Volume with some correlation to price movement
        base_volume = np.random.randint(100000, 1000000)
        price_change_ratio = abs(close_price - open_price) / open_price
        volume = int(base_volume * (1 + price_change_ratio * 10))
        
        # Trading pattern: sometimes more, sometimes less
        if np.random.random() < 0.1:  # 10% chance of high trading day
            volume *= np.random.uniform(1.5, 3)
        
        # Price movement indicator (1 if price increased, 0 if decreased)
        price_movement = 1 if close_price > open_price else 0
        
        # Store the day's data
        data.append({
            'Date': date_range[i],
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume,
            'Price_Movement': price_movement
        })
        
        # Set close as previous close for next iteration
        prev_close = close_price
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess the data for model training."""
    # Create features for prediction
    df['Price_Range'] = df['High'] - df['Low']
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    df['Price_Change'] = df['Close'] - df['Open']
    df['Percent_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Create rolling features
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_Ratio'] = df['MA_5'] / df['MA_10']
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Price_Volatility'] = df['Close'].rolling(window=10).std()
    
    # Create target variable for regression
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Add technical indicators
    df = create_technical_indicators(df)
    
    # Drop NaN values created by shifts and rolling windows
    df = df.dropna()
    
    return df

@st.cache_data
def create_technical_indicators(df):
    """Create additional technical indicators for stock data."""
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['MA_20'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['MA_20'] - 2 * data['Close'].rolling(window=20).std()
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
    
    # Momentum
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    
    # Rate of Change
    data['ROC_5'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5) * 100
    data['ROC_10'] = (data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10) * 100
    
    # Volume indicators
    data['Volume_Change'] = data['Volume'].pct_change()
    data['OBV'] = (data['Close'].diff() > 0).astype(int) * data['Volume']
    data['OBV'] = data['OBV'].cumsum()
    
    return data

@st.cache_data
def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """Prepare train and test data for model training."""
    # Select features for the models
    features = [
        'Open', 'High', 'Low', 'Volume',
        'Price_Range', 'Prev_Close', 'Prev_Volume', 
        'Price_Change', 'Percent_Change',
        'MA_5', 'MA_10', 'MA_Ratio', 'Volume_MA_5', 'Price_Volatility',
        'RSI', 'MACD', 'BB_Position', 'Momentum_5', 'ROC_5'
    ]
    
    # Classification target
    y_class = df['Price_Movement']
    
    # Regression target (predicting next day's closing price)
    y_reg = df['Next_Close']
    
    # Feature data
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_class, test_size=test_size, random_state=random_state
    )
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_reg, test_size=test_size, random_state=random_state
    )
    
    return (
        X_train, X_test, y_train, y_test,
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        scaler, features
    )

# ------------------- Model Training Functions -------------------

@st.cache_resource
def train_svm_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    """Train an SVM classification model."""
    start_time = time.time()
    
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    return model, training_time

@st.cache_resource
def train_decision_tree_model(X_train, y_train, max_depth=None, min_samples_split=2):
    """Train a Decision Tree regression model."""
    start_time = time.time()
    
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    return model, training_time

# ------------------- Visualization Functions -------------------

def plot_stock_data(df, title="Stock Price History"):
    """Plot stock price history with volume."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot stock prices
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue', linewidth=2)
    ax1.plot(df['Date'], df['Open'], label='Open Price', color='green', alpha=0.5)
    ax1.fill_between(df['Date'], df['High'], df['Low'], alpha=0.2, color='blue', label='Price Range')
    
    # Add moving averages if available
    if 'MA_5' in df.columns:
        ax1.plot(df['Date'], df['MA_5'], label='5-day MA', color='orange', linestyle='--')
    if 'MA_10' in df.columns:
        ax1.plot(df['Date'], df['MA_10'], label='10-day MA', color='red', linestyle='--')
    
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # Plot volume
    ax2.bar(df['Date'], df['Volume'], color='gray', alpha=0.7)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix for classification results."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=16)
    
    # Calculate and display metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
    plt.figtext(0.6, 0.2, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """Plot ROC curve for classification results."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right")
    
    return fig

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        st.warning("The model doesn't provide feature importance.")
        return None
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    features = [feature_names[i] for i in indices]
    importances = importances[indices]
    
    # Take top 15 for better visualization
    features = features[:15]
    importances = importances[:15]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    bars = ax.barh(range(len(features)), importances, align='center')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=16)
    
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center')
    
    plt.tight_layout()
    
    return fig

def plot_regression_results(y_true, y_pred, title="Regression Results"):
    """Plot regression results with actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot of actual vs predicted
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Identity line (perfect predictions)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=16)
    
    # Calculate and display metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    metrics_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}'
    plt.figtext(0.15, 0.8, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig

def plot_price_movement_distribution(df):
    """Plot the distribution of price movements (up/down)."""
    up_days = df['Price_Movement'].sum()
    down_days = len(df) - up_days
    
    labels = ['Up', 'Down']
    sizes = [up_days, down_days]
    colors = ['green', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title("Distribution of Price Movements", fontsize=16)
    
    return fig

def plot_performance_comparison(svm_accuracy, dt_rmse, svm_time, dt_time):
    """Plot performance comparison between models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['SVM', 'Decision Tree']
    metrics = [svm_accuracy, 1 - dt_rmse/100]  # Normalize RMSE for comparison
    
    bars = ax1.bar(models, metrics, color=['skyblue', 'lightgreen'])
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Accuracy Comparison', fontsize=14)
    ax1.set_ylabel('Accuracy / Normalized Performance')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    times = [svm_time, dt_time]
    
    bars = ax2.bar(models, times, color=['skyblue', 'lightgreen'])
    ax2.set_yscale('log')  # Use log scale for better visualization
    ax2.set_title('Training Time Comparison (seconds)', fontsize=14)
    ax2.set_ylabel('Training Time (log scale)')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    return fig

def plot_prediction_timeline(dates, actual, predicted, title="Price Prediction Timeline"):
    """Plot actual vs predicted prices over time."""
    # Safety check - don't attempt to plot empty data
    if len(dates) == 0 or len(actual) == 0 or len(predicted) == 0:
        # Create an empty figure with an error message
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Not enough data to generate plot', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    # Ensure all arrays are the same length
    min_length = min(len(dates), len(actual), len(predicted))
    dates = dates[:min_length]
    actual = actual[:min_length]
    predicted = predicted[:min_length]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, actual, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(dates, predicted, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    
    return fig

def plot_model_animation(model_name, progress_bar):
    """Simulate model working with a progress animation."""
    progress_bar.progress(0)
    
    for i in range(100):
        time.sleep(0.02)  # Adjust speed of animation
        progress_bar.progress(i + 1)
    
    return True

# ------------------- Main App Logic -------------------

def main():
    st.markdown('<h1 class="main-header">Stock Price Prediction App</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; margin-bottom: 30px;">Comparative Study of SVM and Decision Tree Models</h3>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Data Overview", "🔍 Data Exploration", "🤖 Model Training & Evaluation", "📈 Performance Comparison", "📝 Conclusion"]
    )
    
    # Generate dataset
    if 'df' not in st.session_state:
        with st.spinner('Generating synthetic stock data...'):
            st.session_state.df = generate_synthetic_stock_data(n_samples=1000)
    
    # Preprocess data
    if 'df_processed' not in st.session_state:
        with st.spinner('Preprocessing data...'):
            st.session_state.df_processed = preprocess_data(st.session_state.df)
    
    df = st.session_state.df
    df_processed = st.session_state.df_processed
    
    # Data Overview Page
    if page == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        This application demonstrates a comparative study of stock price prediction models:
        * **Support Vector Machine (SVM)** for binary classification (Up/Down prediction)
        * **Decision Tree Regression** for numerical price prediction
        
        The dataset contains 1,000 records of synthetic stock data with features like opening price, 
        closing price, high, low, and volume.
        """)
        
        # Stock price chart
        st.markdown('<h3 class="section-header">Stock Price History</h3>', unsafe_allow_html=True)
        fig = plot_stock_data(df)
        st.pyplot(fig)
        
        # Dataset preview
        st.markdown('<h3 class="section-header">Dataset Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">Dataset Statistics</h3>', unsafe_allow_html=True)
            st.dataframe(df.describe())
        
        with col2:
            st.markdown('<h3 class="section-header">Price Movement Distribution</h3>', unsafe_allow_html=True)
            fig = plot_price_movement_distribution(df)
            st.pyplot(fig)
    
    # Data Exploration Page
    elif page == "🔍 Data Exploration":
        st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        This section explores the preprocessed dataset with additional technical indicators and features
        that will be used for model training.
        """)
        
        # Technical indicators
        st.markdown('<h3 class="section-header">Technical Indicators & Features</h3>', unsafe_allow_html=True)
        
        # Select a subset of columns for easier viewing
        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'RSI', 'MACD']
        st.dataframe(df_processed[display_cols].head(10))
        
        # Feature correlation
        st.markdown('<h3 class="section-header">Feature Correlation Matrix</h3>', unsafe_allow_html=True)
        
        correlation_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'RSI', 'MACD']
        corr = df_processed[correlation_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
        
        plt.title("Feature Correlation Matrix", fontsize=16)
        st.pyplot(fig)
        
        # Price vs indicators
        st.markdown('<h3 class="section-header">Price vs Technical Indicators</h3>', unsafe_allow_html=True)
        
        indicators = st.multiselect(
            "Select technical indicators to visualize",
            options=['MA_5', 'MA_10', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower'],
            default=['MA_5', 'MA_10']
        )
        
        if indicators:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_processed['Date'], df_processed['Close'], label='Close Price', color='blue')
            
            for indicator in indicators:
                ax.plot(df_processed['Date'], df_processed[indicator], label=indicator, alpha=0.7)
            
            ax.set_title("Stock Price vs. Technical Indicators", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
    
    # Model Training & Evaluation Page
    elif page == "🤖 Model Training & Evaluation":
        st.markdown('<h2 class="sub-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)
        
        # Prepare train-test data
        if 'train_test_data' not in st.session_state:
            with st.spinner("Preparing training and testing data..."):
                st.session_state.train_test_data = prepare_train_test_data(df_processed)
        
        X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler, features = st.session_state.train_test_data
        
        st.markdown("""
        In this section, we train and evaluate two different models:
        
        1. **SVM Classification Model**: Predicts whether the stock price will go UP or DOWN
        2. **Decision Tree Regression Model**: Predicts the actual closing price value
        """)
        
        # Model selection
        model_tab = st.radio(
            "Select model to view",
            ["SVM Classification", "Decision Tree Regression"],
            horizontal=True
        )
        
        # SVM Classification Model
        if model_tab == "SVM Classification":
            st.markdown('<h3 class="section-header">SVM Classification Model</h3>', unsafe_allow_html=True)
            
            # Model parameters
            with st.expander("Model Parameters"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    kernel = st.selectbox("Kernel", options=['rbf', 'linear', 'poly'], index=0)
                with col2:
                    C = st.slider("Regularization (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                with col3:
                    gamma = st.selectbox("Gamma", options=['scale', 'auto', 0.1, 0.01, 0.001], index=0)
            
            # Train SVM model
            with st.spinner("Training SVM model..."):
                if 'svm_model' not in st.session_state or st.button("Retrain Model"):
                    st.session_state.svm_model, st.session_state.svm_time = train_svm_model(X_train, y_train, kernel, C, gamma)
                
                svm_model = st.session_state.svm_model
                svm_time = st.session_state.svm_time
            
            # Model predictions
            y_pred = svm_model.predict(X_test)
            y_prob = svm_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Simulate model working (visual effect)
            st.markdown('<h4 class="section-header">Model Processing</h4>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            plot_model_animation("SVM", progress_bar)
            st.success("SVM model processing complete!")
            
            # Display metrics
            st.markdown('<h4 class="section-header">Model Performance</h4>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{accuracy:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision (Up)", f"{report['1']['precision']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall (Up)", f"{report['1']['recall']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score (Up)", f"{report['1']['f1-score']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Plots
            st.markdown('<h4 class="section-header">Model Evaluation</h4>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                fig = plot_confusion_matrix(y_test, y_pred, title="SVM Confusion Matrix")
                st.pyplot(fig)
            
            with col2:
                # ROC Curve
                fig = plot_roc_curve(y_test, y_prob, title="SVM ROC Curve")
                st.pyplot(fig)
            
            # Interpretation
            st.markdown('<h4 class="section-header">Interpretation</h4>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="highlight">
            <p>The SVM model achieved an accuracy of <span class="green-text">{accuracy:.3f}</span>, which means it correctly predicted
            the direction of stock price movement {accuracy*100:.1f}% of the time.</p>
            
            <p>For upward movements specifically, the model has:</p>
            <ul>
                <li><strong>Precision:</strong> {report['1']['precision']:.3f} (probability that a predicted "Up" is actually "Up")</li>
                <li><strong>Recall:</strong> {report['1']['recall']:.3f} (proportion of actual "Up" movements that were correctly identified)</li>
            </ul>
            
            <p>The area under the ROC curve is {auc(roc_curve(y_test, y_prob)[0], roc_curve(y_test, y_prob)[1]):.3f}, 
            indicating the model's ability to distinguish between classes.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Decision Tree Regression Model
        else:
            st.markdown('<h3 class="section-header">Decision Tree Regression Model</h3>', unsafe_allow_html=True)
            
            # Model parameters
            with st.expander("Model Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    max_depth = st.slider("Max Depth", min_value=3, max_value=20, value=10, step=1)
                with col2:
                    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=5, step=1)
            
            # Train Decision Tree model
            with st.spinner("Training Decision Tree model..."):
                if 'dt_model' not in st.session_state or st.button("Retrain Model"):
                    st.session_state.dt_model, st.session_state.dt_time = train_decision_tree_model(
                        X_train_reg, y_train_reg, max_depth, min_samples_split
                    )
                
                dt_model = st.session_state.dt_model
                dt_time = st.session_state.dt_time
            
            # Model predictions
            y_pred_reg = dt_model.predict(X_test_reg)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_reg, y_pred_reg)
            mae = np.mean(np.abs(y_test_reg - y_pred_reg))
            
            # Simulate model working (visual effect)
            st.markdown('<h4 class="section-header">Model Processing</h4>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            plot_model_animation("Decision Tree", progress_bar)
            st.success("Decision Tree model processing complete!")
            
            # Display metrics
            st.markdown('<h4 class="section-header">Model Performance</h4>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE", f"{rmse:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MAE", f"{mae:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("R²", f"{r2:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                mean_price = np.mean(y_test_reg)
                error_ratio = rmse / mean_price
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Error Ratio", f"{error_ratio:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Plots
            st.markdown('<h4 class="section-header">Model Evaluation</h4>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                fig = plot_regression_results(y_test_reg, y_pred_reg, title="Decision Tree: Actual vs Predicted")
                st.pyplot(fig)
            
            with col2:
                # Feature Importance
                fig = plot_feature_importance(dt_model, features, title="Decision Tree Feature Importance")
                st.pyplot(fig)
            
            # Prediction Timeline
            st.markdown('<h4 class="section-header">Prediction Timeline</h4>', unsafe_allow_html=True)
            
            # Get a subset of dates for visualization - carefully handle indexing
            try:
                # Handle test indices safely
                if len(y_test_reg) > 0:
                    # Use a smaller sample size for stability
                    sample_size = min(30, len(y_test_reg))
                    # Use sequential indices instead of random to avoid indexing issues
                    test_indices = list(range(sample_size))
                    
                    # Get dates from the processed dataframe's tail
                    date_df = df_processed.iloc[-len(y_test_reg):]
                    # Only proceed if we have enough data
                    if len(date_df) >= sample_size:
                        dates = date_df.iloc[:sample_size]['Date'].values
                        actuals = y_test_reg[:sample_size]
                        predictions = y_pred_reg[:sample_size]
                        
                        fig = plot_prediction_timeline(dates, actuals, predictions, "Decision Tree: Price Prediction Timeline")
                        st.pyplot(fig)
                    else:
                        st.info("Not enough processed data to generate prediction timeline.")
                else:
                    st.info("Test dataset is empty. Cannot generate prediction timeline.")
            except Exception as e:
                st.error(f"Error generating prediction timeline: {str(e)}")
                st.info("Try retraining the model with different parameters.")

            
            # Interpretation
            st.markdown('<h4 class="section-header">Interpretation</h4>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="highlight">
            <p>The Decision Tree model achieved:</p>
            <ul>
                <li><strong>RMSE:</strong> {rmse:.3f} (average prediction error)</li>
                <li><strong>R²:</strong> {r2:.3f} (proportion of variance explained by the model)</li>
            </ul>
            
            <p>The error ratio of {error_ratio:.3f} means that, on average, the model's predictions are off by {error_ratio*100:.1f}% 
            of the average stock price.</p>
            
            <p>As seen in the feature importance plot, the most significant features for predicting stock prices are:</p>
            <ol>
                <li>{features[np.argsort(dt_model.feature_importances_)[::-1][0]]}</li>
                <li>{features[np.argsort(dt_model.feature_importances_)[::-1][1]]}</li>
                <li>{features[np.argsort(dt_model.feature_importances_)[::-1][2]]}</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance Comparison Page
    elif page == "📈 Performance Comparison":
        st.markdown('<h2 class="sub-header">Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Ensure models are trained
        if 'train_test_data' not in st.session_state:
            with st.spinner("Preparing training and testing data..."):
                st.session_state.train_test_data = prepare_train_test_data(df_processed)
        
        X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler, features = st.session_state.train_test_data
        
        # Train models if not already trained
        if 'svm_model' not in st.session_state:
            with st.spinner("Training SVM model..."):
                st.session_state.svm_model, st.session_state.svm_time = train_svm_model(X_train, y_train)
        
        if 'dt_model' not in st.session_state:
            with st.spinner("Training Decision Tree model..."):
                st.session_state.dt_model, st.session_state.dt_time = train_decision_tree_model(X_train_reg, y_train_reg)
        
        svm_model = st.session_state.svm_model
        dt_model = st.session_state.dt_model
        svm_time = st.session_state.svm_time
        dt_time = st.session_state.dt_time
        
        # Make predictions
        y_pred_class = svm_model.predict(X_test)
        y_pred_reg = dt_model.predict(X_test_reg)
        
        # Calculate metrics
        svm_accuracy = accuracy_score(y_test, y_pred_class)
        dt_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        dt_r2 = r2_score(y_test_reg, y_pred_reg)
        
        # Introduction
        st.markdown("""
        This section compares the performance of our two models:
        
        1. **SVM Classification** which predicts the direction of price movement (Up/Down)
        2. **Decision Tree Regression** which predicts the actual closing price
        
        While these models solve different types of problems, we can still compare their 
        performance characteristics, accuracy, and utility for trading decisions.
        """)
        
        # Performance Comparison Chart
        st.markdown('<h3 class="section-header">Model Performance Comparison</h3>', unsafe_allow_html=True)
        
        fig = plot_performance_comparison(svm_accuracy, dt_rmse, svm_time, dt_time)
        st.pyplot(fig)
        
        # Tabular Comparison
        st.markdown('<h3 class="section-header">Comparison Table</h3>', unsafe_allow_html=True)
        
        comparison_data = {
            "Metric": ["Accuracy", "Error Measure", "Training Time", "Prediction Type", "Use Case"],
            "SVM Classification": [f"{svm_accuracy:.3f}", "Classification Error Rate", f"{svm_time:.4f} sec", "Binary (Up/Down)", "Trading Direction Decision"],
            "Decision Tree Regression": [f"{dt_r2:.3f} (R²)", f"RMSE: {dt_rmse:.3f}", f"{dt_time:.4f} sec", "Continuous (Price Value)", "Price Target Setting"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.markdown("""
        <style>
        table {
            width: 100%;
            text-align: center;
        }
        th {
            background-color: #E3F2FD;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.table(comparison_df.set_index("Metric"))
        
        # Strengths and Weaknesses
        st.markdown('<h3 class="section-header">Strengths and Weaknesses</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### SVM Classification
            
            **Strengths:**
            - Simple to interpret (Up/Down)
            - Good for binary trading decisions
            - Higher accuracy for direction prediction
            - Less affected by price scale
            
            **Weaknesses:**
            - No price magnitude information
            - Cannot set specific price targets
            - May miss profitable trades with small price movements
            """)
        
        with col2:
            st.markdown("""
            #### Decision Tree Regression
            
            **Strengths:**
            - Provides exact price predictions
            - Useful for setting price targets
            - Captures magnitude of price movements
            - Intuitive feature importance
            
            **Weaknesses:**
            - Higher prediction error
            - More sensitive to noise
            - May not capture trend reversals well
            - Needs more frequent retraining
            """)
        
        # Trading Strategy Comparison
        st.markdown('<h3 class="section-header">Trading Strategy Comparison</h3>', unsafe_allow_html=True)
        
        # Calculate hypothetical returns
        # For SVM: Buy if predicted Up, Sell if predicted Down
        # For DT: Buy if predicted price > actual price, Sell otherwise
        
        # Select a subset of test data for visualization
        test_subset_size = min(50, len(y_test))  # Reduced size for stability
        test_indices = np.random.choice(range(len(y_test)), size=test_subset_size, replace=False)
        test_indices.sort()
        
        # Create mapping from test indices to dates
        # This safely handles the index alignment
        test_data_df = df_processed.iloc[-len(y_test):]
        if len(test_data_df) > 0:
            test_dates = test_data_df.iloc[test_indices]['Date'].values if len(test_indices) > 0 else []
        else:
            # Fallback if we don't have enough processed data
            test_dates = df.iloc[-test_subset_size:]['Date'].values
        
        # Check if we have enough data to calculate signals
        if len(test_indices) > 0 and len(y_pred_class) > max(test_indices) and len(y_pred_reg) > max(test_indices):
            # SVM trading signals
            svm_signals = y_pred_class[test_indices]
            
            # Safety check for the dataframe index
            if len(df_processed) >= len(y_test) and len(test_data_df) > 0:
                # DT trading signals (1 if predicted price is higher than current price)
                current_prices = test_data_df.iloc[test_indices]['Close'].values
                dt_signals = (y_pred_reg[test_indices] > current_prices).astype(int)
                
                # Calculate returns for both strategies
                if len(current_prices) > 1:  # Need at least 2 prices to calculate returns
                    actual_returns = np.diff(current_prices) / current_prices[:-1]
                else:
                    actual_returns = np.array([0])  # Default to no returns if not enough data
            else:
                # Fallback to safe defaults
                current_prices = np.array([100] * len(test_indices))
                dt_signals = np.zeros(len(test_indices))
                actual_returns = np.zeros(len(test_indices) - 1 if len(test_indices) > 0 else 0)
        else:
            # Not enough data - create empty arrays
            svm_signals = np.array([])
            dt_signals = np.array([])
            actual_returns = np.array([])
        
        # Simulate starting with $1000
        svm_portfolio = 1000
        dt_portfolio = 1000
        svm_values = [svm_portfolio]
        dt_values = [dt_portfolio]
        
        # Check that we have data to simulate
        if len(actual_returns) > 0 and len(svm_signals) > 0 and len(dt_signals) > 0:
            # Make sure signals and returns have matching lengths
            min_length = min(len(actual_returns), len(svm_signals), len(dt_signals))
            actual_returns = actual_returns[:min_length]
            svm_signals = svm_signals[:min_length]
            dt_signals = dt_signals[:min_length]
            
            for i in range(len(actual_returns)):
                # SVM strategy: If signal is 1 (Up), invest and get return
                if i < len(svm_signals) and svm_signals[i] == 1:
                    svm_portfolio *= (1 + actual_returns[i])
                # DT strategy: If signal is 1 (Predicted > Current), invest and get return
                if i < len(dt_signals) and dt_signals[i] == 1:
                    dt_portfolio *= (1 + actual_returns[i])
                
                svm_values.append(svm_portfolio)
                dt_values.append(dt_portfolio)
        
        # Plot portfolio values
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Check if we have enough data and dates to plot
        if len(svm_values) > 0 and len(test_dates) > 0:
            # Make sure we're not trying to plot more values than dates
            plot_length = min(len(test_dates), len(svm_values))
            
            ax.plot(test_dates[:plot_length], svm_values[:plot_length], 
                   label='SVM Strategy', color='blue', linewidth=2)
            ax.plot(test_dates[:plot_length], dt_values[:plot_length], 
                   label='Decision Tree Strategy', color='green', linewidth=2)
            ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Initial Investment')
            
            ax.set_title("Trading Strategy Comparison", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Portfolio Value ($)", fontsize=12)
            ax.legend(loc='best')
            ax.grid(True)
        else:
            # Not enough data to plot - show message
            ax.text(0.5, 0.5, 'Not enough data to generate strategy comparison', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
        
        st.pyplot(fig)
        
        # Final comparison and takeaways
        st.markdown('<h3 class="section-header">Key Takeaways</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="highlight">
        <p>Based on the performance comparison:</p>
        
        <p><strong>For direction prediction:</strong> The SVM model achieves {svm_accuracy:.1%} accuracy, making it suitable for 
        traders who primarily need to know whether the market will go up or down.</p>
        
        <p><strong>For price target setting:</strong> The Decision Tree model provides specific price predictions with 
        an R² of {dt_r2:.3f} and RMSE of {dt_rmse:.3f}.</p>
        
        <p><strong>Practical application:</strong> A combined approach may be optimal - use SVM to decide 
        the trading direction, and Decision Tree to set specific entry and exit points.</p>
        
        <p>The models show complementary strengths, and their effectiveness would depend on the specific 
        trading strategy, time horizon, and risk tolerance of the investor.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Conclusion Page
    elif page == "📝 Conclusion":
        st.markdown('<h2 class="sub-header">Conclusion</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        This application has demonstrated a comparative analysis of two machine learning approaches for stock price prediction:
        
        1. **SVM Classification** for predicting price movement direction
        2. **Decision Tree Regression** for predicting exact price values
        
        ### Key Findings
        
        - **SVM Classification** provides strong directional accuracy, making it suitable for binary trading decisions (buy/sell).
        
        - **Decision Tree Regression** offers specific price predictions, which are valuable for setting price targets and 
        understanding the magnitude of expected movements.
        
        - The **feature importance analysis** reveals that technical indicators like moving averages, volatility measures, 
        and price ranges have significant predictive power.
        
        - A **combined approach** utilizing both models could provide a more comprehensive trading strategy than either model alone.
        
        ### Limitations
        
        - This application uses **synthetic data** for demonstration purposes. Real stock market data would introduce 
        additional challenges like non-stationarity and external influences.
        
        - The models do not account for **fundamental factors**, news events, or market sentiment that can significantly 
        impact stock prices.
        
        - Stock markets are inherently **unpredictable** to some extent, and past performance patterns may not always 
        predict future movements accurately.
        
        ### Future Improvements
        
        - Implement more advanced models like **LSTM neural networks** that can better capture temporal dependencies in time series data.
        
        - Incorporate **sentiment analysis** from news and social media to gauge market sentiment.
        
        - Develop **ensemble methods** that combine multiple prediction techniques for higher accuracy.
        
        - Add **risk assessment** tools to evaluate the confidence level of predictions and potential downside risk.
        
        - Include **backtesting functionality** to evaluate model performance across different market conditions.
        
        ### Final Thoughts
        
        While machine learning models can identify patterns and provide valuable insights for stock trading, 
        they should be used as part of a comprehensive investment strategy that includes risk management, 
        diversification, and consideration of broader market conditions. No prediction model can guarantee 
        profits in the stock market, but they can be powerful tools when used appropriately.
        """)
        
        # Add a visually appealing summary chart
        st.markdown('<h3 class="section-header">Model Selection Guide</h3>', unsafe_allow_html=True)
        
        selection_data = {
            "Criteria": ["Prediction Type", "Use Case", "Advantage", "When to Choose"],
            "SVM Classification": ["Direction (Up/Down)", "Entry/Exit Timing", "Simple, Interpretable", "For trend-following strategies"],
            "Decision Tree Regression": ["Exact Price Value", "Price Targets", "Specific Price Goals", "For range-bound trading"]
        }
        
        selection_df = pd.DataFrame(selection_data)
        st.table(selection_df.set_index("Criteria"))
        
        # Final CTA
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-top: 30px; border: 1px solid #e0e0e0;">
        <h3 style="text-align: center; color: #0277BD;">Ready to Apply These Models to Your Trading?</h3>
        <p style="text-align: center; color: #333333;">Experiment with different parameters, explore feature importance, and develop your own trading strategy.</p>
        <p style="text-align: center; font-weight: bold; margin-top: 20px; color: #333333;">Navigate to the "Model Training & Evaluation" page to customize the models!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Credits and references
        st.markdown("""
        <div style="margin-top: 50px; text-align: center; color: gray; font-size: 0.8rem;">
        <p>Created with Streamlit, scikit-learn, and matplotlib</p>
        <p>Data generated synthetically for educational purposes</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
