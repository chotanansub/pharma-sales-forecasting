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
            
            # Ensure positive values for sales data
            prophet_data['y'] = prophet_data['y'].clip(lower=0.01)
            
            # Remove any rows with missing values
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 10:
                raise ValueError("Insufficient data for Prophet model (need at least 10 observations)")
            
            # Determine seasonality settings based on data length
            data_length = len(prophet_data)
            
            # Initialize Prophet model with appropriate settings
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                daily_seasonality=data_length > 30,  # Only if we have enough data
                weekly_seasonality=data_length > 14,
                yearly_seasonality=data_length > 730,  # Only if we have 2+ years
                interval_width=0.95,
                changepoint_prior_scale=0.05,  # More conservative changepoint detection
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                mcmc_samples=0,  # Use MAP estimation for speed
                uncertainty_samples=1000
            )
            
            # Add custom seasonalities if we have enough data
            if data_length > 30:
                self.model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5,
                    prior_scale=10.0
                )
            
            # Fit the model with error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(prophet_data)
            
            self.logger.info(f"Prophet model fitted with {self.seasonality_mode} seasonality on {len(prophet_data)} records")
            
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.model.predict(future_df)
            
            # Extract predictions and confidence intervals
            predictions = pd.Series(forecast['yhat'].values, index=future_dates)
            lower_ci = pd.Series(forecast['yhat_lower'].values, index=future_dates)
            upper_ci = pd.Series(forecast['yhat_upper'].values, index=future_dates)
            
            # Ensure non-negative predictions for sales data
            predictions = predictions.clip(lower=0)
            lower_ci = lower_ci.clip(lower=0)
            upper_ci = upper_ci.clip(lower=0)
            
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
            eval_length = len(eval_data)
            predictions, _, _ = self.predict(start_date, eval_length)
            
            # Align predictions with evaluation data
            min_length = min(len(predictions), len(eval_data))
            actual = eval_data['sales'].iloc[:min_length].values
            pred = predictions.iloc[:min_length].values
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            
            # MAPE with protection against division by zero
            mape_values = []
            for a, p in zip(actual, pred):
                if abs(a) > 1e-8:  # Avoid division by very small numbers
                    mape_values.append(abs((a - p) / a))
            mape = np.mean(mape_values) * 100 if mape_values else 0
            
            # R-squared
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
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
            return {'model_type': 'Prophet', 'status': 'not_fitted'}
        
        try:
            # Get seasonality information
            seasonalities = {}
            if hasattr(self.model, 'seasonalities'):
                for name, seasonality in self.model.seasonalities.items():
                    seasonalities[name] = {
                        'period': seasonality['period'],
                        'fourier_order': seasonality['fourier_order'],
                        'prior_scale': seasonality['prior_scale']
                    }
            
            return {
                'model_type': 'Prophet',
                'seasonality_mode': self.seasonality_mode,
                'interval_width': 0.95,
                'changepoint_prior_scale': self.model.changepoint_prior_scale,
                'seasonality_prior_scale': self.model.seasonality_prior_scale,
                'n_changepoints': self.model.n_changepoints,
                'seasonalities': seasonalities,
                'uncertainty_samples': getattr(self.model, 'uncertainty_samples', 1000)
            }
            
        except Exception as e:
            self.logger.warning(f"Error getting model info: {str(e)}")
            return {'model_type': 'Prophet', 'status': 'error'}
    
    def get_component_importance(self) -> Dict[str, float]:
        """Get the importance/contribution of different components."""
        if self.model is None:
            return {}
        
        try:
            # This would require fitting the model and analyzing components
            # For now, return basic component information
            components = {
                'trend': 'included',
                'seasonality_mode': self.seasonality_mode,
                'weekly': 'included' if getattr(self.model, 'weekly_seasonality', False) else 'excluded',
                'yearly': 'included' if getattr(self.model, 'yearly_seasonality', False) else 'excluded',
                'daily': 'included' if getattr(self.model, 'daily_seasonality', False) else 'excluded'
            }
            
            return components
            
        except Exception as e:
            self.logger.warning(f"Error getting component importance: {str(e)}")
            return {}
    
    def get_changepoints(self) -> Dict[str, Any]:
        """Get information about detected changepoints."""
        if self.model is None:
            return {}
        
        try:
            changepoints_info = {
                'n_changepoints': self.model.n_changepoints,
                'changepoint_prior_scale': self.model.changepoint_prior_scale,
                'changepoint_range': getattr(self.model, 'changepoint_range', 0.8)
            }
            
            # If model is fitted, get actual changepoints
            if hasattr(self.model, 'changepoints') and self.model.changepoints is not None:
                changepoints_info['changepoints'] = self.model.changepoints.tolist()
            
            return changepoints_info
            
        except Exception as e:
            self.logger.warning(f"Error getting changepoints: {str(e)}")
            return {}