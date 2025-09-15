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
        self.best_order = None
        self.logger = logging.getLogger(__name__)
        
    def check_stationarity(self, data: pd.Series) -> Tuple[bool, float]:
        """Check if the time series is stationary using ADF test."""
        try:
            result = adfuller(data.dropna())
            p_value = result[1]
            is_stationary = p_value <= 0.05
            return is_stationary, p_value
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {str(e)}")
            return False, 1.0
        
    def find_optimal_order(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using grid search with AIC."""
        try:
            data = data.dropna()
            if len(data) < 10:
                return (1, 1, 1)
            
            # Check stationarity to determine d
            is_stationary, p_value = self.check_stationarity(data)
            
            # Start with d=0 if stationary, otherwise try d=1
            d_values = [0] if is_stationary else [1]
            
            # If series is very non-stationary, also try d=2
            if p_value > 0.1:
                d_values.append(2)
            
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Grid search ranges (smaller for efficiency)
            p_range = range(0, min(4, len(data)//10 + 1))
            q_range = range(0, min(4, len(data)//10 + 1))
            
            for d in d_values:
                for p in p_range:
                    for q in q_range:
                        try:
                            # Skip some invalid combinations
                            if p == 0 and d == 0 and q == 0:
                                continue
                                
                            model = ARIMA(data, order=(p, d, q))
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                
                        except Exception:
                            continue
            
            self.best_order = best_order
            self.logger.info(f"Optimal ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            return best_order
            
        except Exception as e:
            self.logger.warning(f"Error in order selection, using default (1,1,1): {str(e)}")
            self.best_order = (1, 1, 1)
            return (1, 1, 1)
    
    def fit(self, data: pd.Series) -> None:
        """Fit ARIMA model to the data."""
        try:
            # Clean data
            data = data.dropna()
            if len(data) < 10:
                raise ValueError("Insufficient data for ARIMA model (need at least 10 observations)")
            
            # Ensure positive values for sales data
            data = data.clip(lower=0.01)  # Avoid zeros which can cause issues
            
            # Find optimal order
            order = self.find_optimal_order(data)
            
            # Fit model
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
            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            confidence_int = forecast_result.conf_int()
            
            # Create series with proper index
            predictions = pd.Series(forecast.values, index=range(len(forecast)))
            lower_ci = pd.Series(confidence_int.iloc[:, 0].values, index=range(len(forecast)))
            upper_ci = pd.Series(confidence_int.iloc[:, 1].values, index=range(len(forecast)))
            
            # Ensure non-negative predictions for sales data
            predictions = predictions.clip(lower=0)
            lower_ci = lower_ci.clip(lower=0)
            upper_ci = upper_ci.clip(lower=0)
            
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
            eval_length = len(eval_data)
            predictions, _, _ = self.predict(eval_length)
            
            # Align predictions with evaluation data
            min_length = min(len(predictions), len(eval_data))
            actual = eval_data.iloc[:min_length].values
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
            self.logger.error(f"Error in ARIMA evaluation: {str(e)}")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        if self.fitted_model is None:
            return {'model_type': 'ARIMA', 'status': 'not_fitted'}
        
        try:
            return {
                'model_type': 'ARIMA',
                'order': self.fitted_model.model.order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'seasonal_periods': self.seasonal_periods,
                'params': len(self.fitted_model.params),
                'loglikelihood': self.fitted_model.llf
            }
        except Exception as e:
            self.logger.warning(f"Error getting model info: {str(e)}")
            return {'model_type': 'ARIMA', 'status': 'error'}
    
    def get_residual_diagnostics(self) -> Dict[str, Any]:
        """Get residual diagnostics for model validation."""
        if self.fitted_model is None:
            return {}
        
        try:
            residuals = self.fitted_model.resid
            
            # Ljung-Box test for residual autocorrelation
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            diagnostics = {
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
                'residual_autocorr': 'passed' if ljung_box['lb_pvalue'].iloc[-1] > 0.05 else 'failed'
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.warning(f"Error in residual diagnostics: {str(e)}")
            return {}