import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

class CatBoostModel:
    def __init__(self, iterations: int = 1000, learning_rate: float = 0.1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for time series forecasting."""
        try:
            features_df = data.copy()
            
            # Time-based features
            features_df['year'] = features_df['date'].dt.year
            features_df['month'] = features_df['date'].dt.month
            features_df['day'] = features_df['date'].dt.day
            features_df['dayofweek'] = features_df['date'].dt.dayofweek
            features_df['dayofyear'] = features_df['date'].dt.dayofyear
            features_df['quarter'] = features_df['date'].dt.quarter
            
            # Lag features
            for lag in [1, 2, 3, 7, 14, 30]:
                features_df[f'lag_{lag}'] = features_df['sales'].shift(lag)
            
            # Rolling statistics
            for window in [3, 7, 14, 30]:
                features_df[f'rolling_mean_{window}'] = features_df['sales'].rolling(window=window).mean()
                features_df[f'rolling_std_{window}'] = features_df['sales'].rolling(window=window).std()
            
            # Trend features
            features_df['sales_diff_1'] = features_df['sales'].diff(1)
            features_df['sales_diff_7'] = features_df['sales'].diff(7)
            
            # Drop the original date column for modeling
            feature_columns = [col for col in features_df.columns if col not in ['date', 'sales']]
            
            return features_df[feature_columns + ['sales']]
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            raise
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit CatBoost model to the data."""
        try:
            # Create features
            features_data = self.create_features(data)
            features_data = features_data.dropna()  # Remove NaN values from lag features
            
            if len(features_data) == 0:
                raise ValueError("No data available after feature creation and NaN removal")
            
            X = features_data.drop('sales', axis=1)
            y = features_data['sales']
            
            # Initialize and fit CatBoost model
            self.model = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=6,
                loss_function='RMSE',
                random_state=42,
                verbose=False
            )
            
            self.model.fit(X, y)
            
            self.logger.info(f"CatBoost model fitted with {self.iterations} iterations")
            
        except Exception as e:
            self.logger.error(f"Error fitting CatBoost model: {str(e)}")
            raise
    
    def predict(self, train_data: pd.DataFrame, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            predictions = []
            current_data = train_data.copy()
            
            # Generate predictions iteratively
            for _ in range(steps):
                # Create features for the current data
                features_data = self.create_features(current_data)
                features_data = features_data.dropna()
                
                if len(features_data) == 0:
                    break
                    
                X_pred = features_data.drop('sales', axis=1).iloc[-1:].values
                pred = self.model.predict(X_pred)[0]
                predictions.append(pred)
                
                # Add prediction to data for next iteration
                next_date = current_data['date'].iloc[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame({
                    'date': [next_date],
                    'sales': [pred]
                })
                current_data = pd.concat([current_data, new_row], ignore_index=True)
            
            # Create date range for predictions
            start_date = train_data['date'].iloc[-1] + pd.Timedelta(days=1)
            pred_dates = pd.date_range(start=start_date, periods=len(predictions), freq='D')
            
            pred_series = pd.Series(predictions, index=pred_dates)
            
            # For CatBoost, we'll use a simple confidence interval based on training residuals
            if len(predictions) > 0:
                # Calculate training residuals to estimate confidence intervals
                train_features = self.create_features(train_data).dropna()
                X_train = train_features.drop('sales', axis=1)
                y_train = train_features['sales']
                train_pred = self.model.predict(X_train)
                residuals_std = np.std(y_train - train_pred)
                
                lower_ci = pred_series - 1.96 * residuals_std
                upper_ci = pred_series + 1.96 * residuals_std
            else:
                lower_ci = pred_series
                upper_ci = pred_series
            
            return pred_series, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in CatBoost prediction: {str(e)}")
            raise
    
    def evaluate(self, train_data: pd.DataFrame, eval_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Fit on training data
            self.fit(train_data)
            
            # Predict evaluation period
            predictions, _, _ = self.predict(train_data, len(eval_data))
            
            if len(predictions) == 0:
                return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
            
            # Calculate metrics
            actual = eval_data['sales'].values[:len(predictions)]
            pred = predictions.values[:len(actual)]
            
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            # R-squared
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
        except Exception as e:
            self.logger.error(f"Error in CatBoost evaluation: {str(e)}")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        if self.model is None:
            return {}
        
        return {
            'model_type': 'CatBoost',
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': 6,
            'loss_function': 'RMSE'
        }