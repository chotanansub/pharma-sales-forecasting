import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import logging
from typing import Tuple, Dict, Any

warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        self.logger = logging.getLogger(__name__)
        
    def find_optimal_order(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using grid search."""
        try:
            # Check stationarity
            adf_result = adfuller(data.dropna())
            is_stationary = adf_result[1] <= 0.05
            d = 0 if is_stationary else 1
            
            # Grid search for p and q
            best_aic = float('inf')
            best_order = (1, d, 1)
            
            for p in range(0, 4):
                for q in range(0, 4):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
            
            self.logger.info(f"Optimal ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            return best_order
            
        except Exception as e:
            self.logger.warning(f"Error in order selection, using default (1,1,1): {str(e)}")
            return (1, 1, 1)
    
    def fit(self, data: pd.Series) -> None:
        """Fit ARIMA model to the data."""
        try:
            order = self.find_optimal_order(data)
            self.model = ARIMA(data, order=order)
            self.fitted_model = self.model.fit()
            
            self.logger.info(f"ARIMA model fitted with order {order}")
            
        except Exception as e:
            self.logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
    
    def predict(self, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate predictions with confidence intervals."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            confidence_int = self.fitted_model.get_forecast(steps=steps).conf_int()
            
            predictions = pd.Series(forecast)
            lower_ci = pd.Series(confidence_int.iloc[:, 0])
            upper_ci = pd.Series(confidence_int.iloc[:, 1])
            
            return predictions, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA prediction: {str(e)}")
            raise
    
    def evaluate(self, train_data: pd.Series, eval_data: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Fit on training data
            self.fit(train_data)
            
            # Predict evaluation period
            predictions, _, _ = self.predict(len(eval_data))
            
            # Calculate metrics
            mae = np.mean(np.abs(eval_data.values - predictions.values))
            rmse = np.sqrt(np.mean((eval_data.values - predictions.values) ** 2))
            mape = np.mean(np.abs((eval_data.values - predictions.values) / eval_data.values)) * 100
            
            # R-squared
            ss_res = np.sum((eval_data.values - predictions.values) ** 2)
            ss_tot = np.sum((eval_data.values - np.mean(eval_data.values)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA evaluation: {str(e)}")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        if self.fitted_model is None:
            return {}
        
        return {
            'model_type': 'ARIMA',
            'order': self.fitted_model.model.order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'seasonal_periods': self.seasonal_periods
        }