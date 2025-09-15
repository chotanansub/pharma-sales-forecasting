# Pharmacy Sales Forecasting System

A comprehensive pharmacy sales prediction system that processes historical daily sales data and generates accurate forecasts using multiple machine learning models. **Specially optimized for intermittent demand patterns** commonly found in pharmaceutical data with many zero-sales periods.

## 🚀 Key Features

- **Multi-Model Forecasting**: ARIMA, Prophet, and CatBoost models for robust predictions
- **Intermittent Demand Specialization**: Handling of sparse/zero-inflated pharmacy sales data
- **Automated Data Processing**: Seamlessly combines monthly CSV files with data validation
- **Comprehensive Evaluation**: Multiple metrics including intermittent-specific performance indicators
- **Flexible Configuration**: Customizable forecast periods, model parameters, and output formats

## 📁 Project Structure

```
pharmacy-sales-forecasting/
├── data/
│   └── sales/
│       ├── sales_2014_01.csv    # January 2014 data
│       ├── sales_2014_02.csv    # February 2014 data
│       ├── sales_2014_03.csv    # March 2014 data
│       ├── sales_2014_04.csv    # April 2014 data
│       ├── sales_2014_05.csv    # May 2014 data
│       ├── sales_2014_06.csv    # June 2014 data
│       ├── sales_2014_07.csv    # July 2014 data
│       ├── sales_2014_08.csv    # August 2014 data
│       └── ...                  # Additional monthly files
├── src/
│   ├── main.py                 # Main entry point
│   ├── config.py              # Configuration settings
│   ├── data_processor.py      # Data loading and processing
│   ├── forecaster.py          # Main forecasting orchestrator
│   └── models/
│       ├── arima_model.py     # ARIMA implementation
│       ├── prophet_model.py   # Prophet implementation
│       └── catboost_model.py  # CatBoost with intermittent demand features
├── predictions/               # Output directory
│   └── sales_MM_YYYY/        # Results organized by latest month
│       ├── ARIMA/
│       ├── Prophet/
│       └── CatBoost/
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd pharmacy-sales-forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
- Place CSV files in `data/sales/` folder
- Follow naming convention: `sales_YYYY_MM.csv` (e.g., `sales_2014_08.csv`)
- Ensure CSV format matches the expected schema (see Data Format section)

## 📊 Data Format Requirements

### File Naming Convention
```
sales_YYYY_MM.csv
```
**Examples**: `sales_2014_01.csv`, `sales_2014_08.csv`, `sales_2015_12.csv`

### CSV File Structure
```csv
date,M01AB,M01AE,N02BA,N02BE,N05B,N05C,R03,R06
2014-08-01,2.0,1.0,2.2,14.0,17.0,1.0,0.0,1.0
2014-08-02,5.0,2.01,6.0,29.0,12.0,0.0,5.0,3.0
2014-08-03,5.0,1.0,3.0,6.0,5.0,5.0,0.0,0.0
...
```

**Required columns**:
- `date`: Date in YYYY-MM-DD format
- Drug columns: Any number of pharmaceutical product columns with sales values

**Optional columns** (automatically excluded from analysis):
- `source_file`, `Year`, `Month`, `Hour`, `Weekday Name`

## 🚀 Usage

### Quick Start
```bash
# Run with default settings (180-day forecast for all drugs)
python src/main.py
```

### Command Line Options

**Custom forecast period**:
```bash
python src/main.py --forecast_days 365
```

**Analyze specific drugs**:
```bash
python src/main.py --drug_filter M01AB,N02BA,N05B
```

**Custom data and output paths**:
```bash
python src/main.py --data_path /path/to/sales/files/ --output_path /path/to/results/
```

**All options combined**:
```bash
python src/main.py \
    --forecast_days 180 \
    --drug_filter M01AB,N02BA \
    --output_path predictions/ \
    --data_path data/sales/ \
    --train_ratio 0.7 \
    --log_level INFO
```

### Available Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--forecast_days` | Number of days to forecast | 180 |
| `--drug_filter` | Comma-separated list of drugs to analyze | All drugs |
| `--output_path` | Output directory path | `predictions/` |
| `--data_path` | Input data directory path | `data/sales/` |
| `--train_ratio` | Training data ratio for evaluation | 0.7 |
| `--log_level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |

### Programmatic Usage
```python
from src.forecaster import Forecaster

# Initialize forecaster
forecaster = Forecaster(forecast_days=180)

# Load data from custom path
forecaster.load_data('data/sales/')

# Run complete analysis
results = forecaster.run_complete_analysis(drug_filter=['M01AB', 'N02BA'])

# Access results for specific drug
drug_results = results['M01AB']
evaluation_metrics = drug_results['evaluation_results']
predictions = drug_results['predictions']

# Get stored results from last analysis
last_results = forecaster.get_results()
```

## 📈 Forecasting Models

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Automatic order selection** using AIC optimization
- **Stationarity testing** with ADF test
- **Seasonal pattern detection**
- **Best for**: Linear trends and classical time series patterns
- **Handles**: Regular demand patterns with clear seasonality

### 2. Prophet (Facebook's Time Series Forecasting)
- **Automatic seasonality detection** (daily, weekly, yearly)
- **Trend changepoint detection**
- **Holiday effects modeling**
- **Robust uncertainty intervals**
- **Best for**: Data with strong seasonal patterns and trend changes
- **Handles**: Regular to moderately irregular demand

### 3. CatBoost (Gradient Boosting) - **Intermittent Demand Specialist**
- **Automatic intermittent pattern detection**
- **Specialized feature engineering** for sparse data
- **Adaptive loss functions** (Poisson for highly sparse data)
- **Sample weighting** for zero-inflated distributions
- **Croston's method integration** for demand size and interval modeling
- **Advanced metrics** for intermittent demand evaluation
- **Best for**: Intermittent demand, zero-inflated data, complex patterns

## 🎯 Intermittent Demand Specialization

### What is Intermittent Demand?
Intermittent demand occurs when products have:
- Many periods with zero sales (25%+ of time periods)
- Unpredictable and sporadic demand patterns
- High variability in both timing and quantity
- Common in specialty drugs, seasonal medications, rare treatments

### Automatic Detection and Handling
The system automatically detects intermittent patterns and applies specialized techniques:

- **Regular demand** (<25% zeros): Standard model configuration
- **Intermittent demand** (25-60% zeros): Enhanced features + regularization
- **Highly intermittent** (>60% zeros): Poisson loss + sample weighting

### Specialized Features for Sparse Data

#### 1. **Demand Pattern Analysis**
- Intermittent ratio calculation
- Average demand intervals
- Demand size variability analysis
- Seasonal intermittency detection

#### 2. **Advanced Feature Engineering**
- **Time-based**: Time since last demand, consecutive zero periods
- **Probability**: Rolling demand occurrence probabilities
- **Croston's components**: Separate demand size and interval modeling
- **Binary indicators**: Demand occurrence, seasonal patterns
- **Exponential smoothing**: Adaptive smoothing for sparse patterns

#### 3. **Model Adaptations**
- **Loss functions**: Poisson for count-like sparse data
- **Sample weighting**: Higher weight for non-zero observations
- **Regularization**: Increased to prevent overfitting to zeros
- **Post-processing**: Threshold-based zero assignment

#### 4. **Specialized Evaluation Metrics**
- **Zero Accuracy**: Accuracy of zero-demand period predictions
- **Demand Occurrence Accuracy**: Binary classification of demand vs no-demand
- **Non-zero MAE**: Performance on periods with actual demand
- **Standard metrics**: MAE, RMSE, MAPE, R² for overall assessment

## 📊 Output Structure

### Directory Organization
```
predictions/
└── sales_MM_YYYY/  # Latest month from input data
    ├── ARIMA/
    │   ├── drugname_arima_MM_YYYY.csv
    │   ├── drugname_arima_MM_YYYY.png
    │   └── drugname_arima_MM_YYYY.txt
    ├── Prophet/
    │   ├── drugname_prophet_MM_YYYY.csv
    │   ├── drugname_prophet_MM_YYYY.png
    │   └── drugname_prophet_MM_YYYY.txt
    └── CatBoost/
        ├── drugname_catboost_MM_YYYY.csv
        ├── drugname_catboost_MM_YYYY.png
        └── drugname_catboost_MM_YYYY.txt
```

### File Contents

#### 1. Prediction Files (CSV)
```csv
date,drug_name,predicted_sales,confidence_lower,confidence_upper
2024-09-16,M01AB,145.2,130.5,159.8
2024-09-17,M01AB,148.1,132.4,163.7
...
```

#### 2. Visualizations (PNG)
- **Multi-panel dashboards** with:
  - Historical data vs predictions with confidence intervals
  - Model performance metrics visualization
  - Sales distribution analysis
  - Monthly trend patterns

#### 3. Evaluation Reports (TXT)
```txt
Model: CatBoost
Drug: M01AB
Training Period: 2014-01-01 to 2014-08-31
Forecast Period: 180 days
Generated: 2024-09-15 10:30:45

Demand Pattern Analysis:
- Intermittent ratio: 45.7% zeros
- Classification: Intermittent demand

Evaluation Metrics:
- MAE: 2.34
- RMSE: 4.12
- MAPE: 12.5%
- R²: 0.73
- Zero Accuracy: 87.3%
- Demand Occurrence Accuracy: 82.1%
- Non-zero MAE: 3.45

Model Parameters:
- iterations: 1000
- learning_rate: 0.07
- loss_function: RMSE
- intermittent_optimization: enabled
```

## 🔍 Model Selection Guidelines

| Data Pattern | Zeros % | Recommended Model | Reason |
|--------------|---------|-------------------|---------|
| Regular demand | <25% | ARIMA or Prophet | Classical time series methods |
| Intermittent demand | 25-60% | **CatBoost** | **Specialized intermittent features** |
| Highly sparse | >60% | **CatBoost** | **Poisson loss + sample weighting** |
| Strong seasonality | Any | Prophet | Best seasonality handling |
| Limited data | Any | Prophet | Robust with small datasets |
| Non-linear patterns | Any | CatBoost | Complex relationship modeling |

### Sample Analysis Output
```
Drug: M01AB
Data points: 247
Intermittent demand: 45.7% zeros
CatBoost - MAE: 2.34, RMSE: 4.12, Zero Acc: 87.3%, Demand Occ Acc: 82.1%

Drug: N02BA
Data points: 247
Highly intermittent demand: 67.2% zeros
CatBoost - MAE: 1.12, RMSE: 2.89, Zero Acc: 92.1%, Demand Occ Acc: 88.4%
```

## ⚙️ Configuration

### Core Parameters (config.py)
```python
# Forecasting settings
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
OUTPUT_BASE_PATH = 'predictions/'
FIGURE_SIZE = (12, 8)
DPI = 300
```

## 🔄 Complete Workflow

1. **Data Discovery**: Scan and validate CSV files in sales directory
2. **Data Combination**: Merge monthly files into unified time series
3. **Quality Validation**: Clean data and handle missing values
4. **Pattern Analysis**: Detect intermittent demand characteristics
5. **Model Training**: Train all three models with pattern-specific optimizations
6. **Performance Evaluation**: Calculate comprehensive metrics including intermittent-specific ones
7. **Prediction Generation**: Create forecasts using full historical data
8. **Output Creation**: Generate CSV files, visualizations, and detailed reports

## 📊 Performance Expectations

### Typical Results by Demand Pattern

| Pattern Type | Zero Accuracy | Demand Occ Accuracy | MAE Range | Best Model |
|-------------|---------------|---------------------|-----------|------------|
| Regular | N/A | N/A | 2-8 | ARIMA/Prophet |
| Intermittent | 80-90% | 75-85% | 1-5 | CatBoost |
| Highly Sparse | 85-95% | 80-90% | 0.5-3 | CatBoost |
