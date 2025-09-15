import os
from datetime import datetime

# Data configuration
DATA_FOLDER = 'data/sales/'
OUTPUT_BASE_PATH = 'predictions/'

# Forecasting parameters
FORECAST_DAYS = 180
TRAIN_TEST_SPLIT = 0.7

# Model parameters
ARIMA_SEASONAL_PERIODS = 7
PROPHET_SEASONALITY_MODE = 'multiplicative'
CATBOOST_ITERATIONS = 1000
CATBOOST_LEARNING_RATE = 0.1

# Intermittent demand settings
INTERMITTENT_ZERO_THRESHOLD = 0.01
INTERMITTENT_RATIO_THRESHOLD = 0.25
HIGHLY_INTERMITTENT_THRESHOLD = 0.6
CROSTON_ALPHA = 0.3
SAMPLE_WEIGHT_RATIO = 3.0

# Output settings
PLOT_STYLE = 'seaborn-v0_8'  # Updated for newer matplotlib versions
FIGURE_SIZE = (12, 8)
DPI = 300

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)