import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

class ProphetModel:
    def __init__(self, seasonality_mode: str = 'multiplicative'):
        self.seasonality_mode = seasonality_mode
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, data: pd.DataFrame) -> None:
        """Fit Prophet model to the data."""
        try:
            # Prepare data for Prophet
            prophet_data = data.copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize Prophet model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
            
            # Fit the model
            self.model.fit(prophet_data)
            
            self.logger.info(f"Prophet model fitted with {self.seasonality_mode} seasonality")
            
        except Exception as e:
            self.logger.error(f"Error fitting Prophet model: {str(e)}")
            raise
    
    def predict(self, start_date: pd.Timestamp, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate predictions with confidence intervals."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Create future dataframe
            future_dates = pd.date_range(start=start_date, periods=steps, freq='D')
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Make predictions
            forecast = self.model.predict(future_df)
            
            predictions = pd.Series(forecast['yhat'].values, index=future_dates)
            lower_ci = pd.Series(forecast['yhat_lower'].values, index=future_dates)
            upper_ci = pd.Series(forecast['yhat_upper'].values, index=future_dates)
            
            return predictions, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in Prophet prediction: {str(e)}")
            raise
    
    def evaluate(self, train_data: pd.DataFrame, eval_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Fit on training data
            self.fit(train_data)
            
            # Predict evaluation period
            start_date = eval_data['date'].iloc[0]
            predictions, _, _ = self.predict(start_date, len(eval_data))
            
            # Calculate metrics
            actual = eval_data['sales'].values
            pred = predictions.values
            
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            # R-squared
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
        except Exception as e:
            self.logger.error(f"Error in Prophet evaluation: {str(e)}")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        if self.model is None:
            return {}
        
        return {
            'model_type': 'Prophet',
            'seasonality_mode': self.seasonality_mode,
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'interval_width': 0.95
        }