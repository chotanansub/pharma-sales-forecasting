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
        self.feature_names = None
        self.zero_threshold = 0.01  # Threshold to consider as zero demand
        self.intermittent_ratio = 0.0  # Ratio of zero/near-zero values
        self.logger = logging.getLogger(__name__)
        
    def analyze_demand_pattern(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze demand pattern to detect intermittency."""
        try:
            total_periods = len(data)
            zero_periods = (data <= self.zero_threshold).sum()
            self.intermittent_ratio = zero_periods / total_periods if total_periods > 0 else 0
            
            # Calculate demand intervals (time between non-zero demands)
            non_zero_indices = data[data > self.zero_threshold].index
            intervals = []
            if len(non_zero_indices) > 1:
                intervals = np.diff(non_zero_indices)
            
            avg_interval = np.mean(intervals) if intervals else 0
            cv_interval = np.std(intervals) / np.mean(intervals) if len(intervals) > 0 and np.mean(intervals) > 0 else 0
            
            # Calculate demand size variability
            non_zero_demands = data[data > self.zero_threshold]
            cv_demand = non_zero_demands.std() / non_zero_demands.mean() if len(non_zero_demands) > 0 and non_zero_demands.mean() > 0 else 0
            
            pattern_info = {
                'total_periods': total_periods,
                'zero_periods': zero_periods,
                'intermittent_ratio': self.intermittent_ratio,
                'avg_demand_interval': avg_interval,
                'cv_interval': cv_interval,
                'cv_demand_size': cv_demand,
                'is_intermittent': self.intermittent_ratio > 0.25,  # More than 25% zeros
                'is_highly_intermittent': self.intermittent_ratio > 0.6,  # More than 60% zeros
                'mean_non_zero_demand': non_zero_demands.mean() if len(non_zero_demands) > 0 else 0,
                'max_demand': data.max(),
                'demand_spikes': (data > data.quantile(0.95)).sum() if len(data) > 0 else 0
            }
            
            self.logger.info(f"Demand pattern analysis: {self.intermittent_ratio:.1%} intermittent, "
                           f"avg interval: {avg_interval:.1f}, CV: {cv_demand:.2f}")
            
            return pattern_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing demand pattern: {str(e)}")
            return {'is_intermittent': False, 'intermittent_ratio': 0}
    
    def create_intermittent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create specialized features for intermittent demand forecasting."""
        try:
            features_df = data.copy()
            
            # Ensure we have a datetime index
            if 'date' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['date'])
            
            # Basic time features
            features_df['year'] = features_df['date'].dt.year
            features_df['month'] = features_df['date'].dt.month
            features_df['day'] = features_df['date'].dt.day
            features_df['dayofweek'] = features_df['date'].dt.dayofweek
            features_df['dayofyear'] = features_df['date'].dt.dayofyear
            features_df['quarter'] = features_df['date'].dt.quarter
            features_df['is_weekend'] = (features_df['date'].dt.dayofweek >= 5).astype(int)
            features_df['is_month_start'] = features_df['date'].dt.is_month_start.astype(int)
            features_df['is_month_end'] = features_df['date'].dt.is_month_end.astype(int)
            
            # Intermittent demand specific features
            sales = features_df['sales']
            
            # Binary demand indicator
            features_df['has_demand'] = (sales > self.zero_threshold).astype(int)
            
            # Time since last demand
            features_df['time_since_demand'] = 0
            last_demand_idx = -1
            for i in range(len(features_df)):
                if sales.iloc[i] > self.zero_threshold:
                    last_demand_idx = i
                    features_df.loc[i, 'time_since_demand'] = 0
                else:
                    features_df.loc[i, 'time_since_demand'] = i - last_demand_idx if last_demand_idx >= 0 else i + 1
            
            # Time until next demand (look-ahead feature for training)
            features_df['time_until_demand'] = 0
            next_demand_idx = len(features_df)
            for i in range(len(features_df) - 1, -1, -1):
                if sales.iloc[i] > self.zero_threshold:
                    next_demand_idx = i
                    features_df.loc[i, 'time_until_demand'] = 0
                else:
                    features_df.loc[i, 'time_until_demand'] = next_demand_idx - i if next_demand_idx < len(features_df) else len(features_df) - i
            
            # Demand intensity features
            lag_periods = [1, 2, 3, 7, 14, 30]
            for lag in lag_periods:
                if len(features_df) > lag:
                    # Lagged demand values
                    features_df[f'lag_demand_{lag}'] = sales.shift(lag)
                    # Lagged binary indicators
                    features_df[f'lag_has_demand_{lag}'] = features_df['has_demand'].shift(lag)
            
            # Rolling demand characteristics
            rolling_windows = [3, 7, 14, 30]
            for window in rolling_windows:
                if len(features_df) > window:
                    # Probability of demand in window
                    features_df[f'demand_prob_{window}'] = features_df['has_demand'].rolling(window=window, min_periods=1).mean()
                    
                    # Average demand when non-zero
                    non_zero_sales = sales.where(sales > self.zero_threshold)
                    features_df[f'avg_nonzero_demand_{window}'] = non_zero_sales.rolling(window=window, min_periods=1).mean()
                    
                    # Demand frequency
                    features_df[f'demand_frequency_{window}'] = features_df['has_demand'].rolling(window=window, min_periods=1).sum()
                    
                    # Coefficient of variation for demand size
                    rolling_mean = sales.rolling(window=window, min_periods=1).mean()
                    rolling_std = sales.rolling(window=window, min_periods=1).std()
                    features_df[f'demand_cv_{window}'] = rolling_std / (rolling_mean + 1e-8)
                    
                    # Maximum demand in period
                    features_df[f'max_demand_{window}'] = sales.rolling(window=window, min_periods=1).max()
            
            # Exponential smoothing for intermittent patterns
            # Simple exponential smoothing for demand occurrence
            alpha = 0.3
            features_df['ema_demand_prob'] = features_df['has_demand'].ewm(alpha=alpha).mean()
            
            # Double exponential smoothing for demand size
            alpha_level = 0.3
            alpha_trend = 0.1
            level = non_zero_sales = sales.where(sales > self.zero_threshold, 0)
            trend = 0
            smoothed_level = []
            smoothed_trend = []
            
            for i in range(len(features_df)):
                if i == 0:
                    level = sales.iloc[i]
                    trend = 0
                else:
                    prev_level = level
                    level = alpha_level * sales.iloc[i] + (1 - alpha_level) * (level + trend)
                    trend = alpha_trend * (level - prev_level) + (1 - alpha_trend) * trend
                
                smoothed_level.append(level)
                smoothed_trend.append(trend)
            
            features_df['smoothed_level'] = smoothed_level
            features_df['smoothed_trend'] = smoothed_trend
            
            # Croston's method components
            # Separate modeling of demand size and intervals
            demand_sizes = []
            demand_intervals = []
            last_nonzero_idx = -1
            
            for i in range(len(sales)):
                if sales.iloc[i] > self.zero_threshold:
                    demand_sizes.append(sales.iloc[i])
                    if last_nonzero_idx >= 0:
                        demand_intervals.append(i - last_nonzero_idx)
                    last_nonzero_idx = i
                else:
                    demand_sizes.append(0)
                    demand_intervals.append(0)
            
            # Exponential smoothing for demand size and intervals
            if demand_sizes and any(d > 0 for d in demand_sizes):
                features_df['croston_demand_size'] = pd.Series(demand_sizes).ewm(alpha=0.3).mean()
            else:
                features_df['croston_demand_size'] = 0
                
            if demand_intervals and any(d > 0 for d in demand_intervals):
                features_df['croston_interval'] = pd.Series(demand_intervals).ewm(alpha=0.3).mean()
            else:
                features_df['croston_interval'] = 1
            
            # Seasonal intermittent patterns
            features_df['sin_dayofyear'] = np.sin(2 * np.pi * features_df['dayofyear'] / 365.25)
            features_df['cos_dayofyear'] = np.cos(2 * np.pi * features_df['dayofyear'] / 365.25)
            features_df['sin_dayofweek'] = np.sin(2 * np.pi * features_df['dayofweek'] / 7)
            features_df['cos_dayofweek'] = np.cos(2 * np.pi * features_df['dayofweek'] / 7)
            features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
            
            # Interaction features for intermittent demand
            features_df['weekend_demand_prob'] = features_df['is_weekend'] * features_df.get('demand_prob_7', 0)
            features_df['month_start_demand'] = features_df['is_month_start'] * features_df.get('avg_nonzero_demand_30', 0)
            
            # Zero-inflated features
            features_df['consecutive_zeros'] = 0
            zero_count = 0
            for i in range(len(sales)):
                if sales.iloc[i] <= self.zero_threshold:
                    zero_count += 1
                else:
                    zero_count = 0
                features_df.loc[i, 'consecutive_zeros'] = zero_count
            
            # Feature scaling for intermittent data
            # Log transform for highly skewed demand
            features_df['log_demand_plus_one'] = np.log1p(sales)
            
            # Square root transform for moderate skewness
            features_df['sqrt_demand'] = np.sqrt(sales)
            
            # Drop the original date column for modeling
            feature_columns = [col for col in features_df.columns if col not in ['date', 'sales']]
            
            # Store feature names for later use
            self.feature_names = feature_columns
            
            return features_df[feature_columns + ['sales']]
            
        except Exception as e:
            self.logger.error(f"Error creating intermittent features: {str(e)}")
            raise
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit CatBoost model optimized for intermittent demand."""
        try:
            # Analyze demand pattern first
            demand_analysis = self.analyze_demand_pattern(data['sales'])
            
            # Create specialized features for intermittent demand
            features_data = self.create_intermittent_features(data)
            
            # Remove rows with NaN values (from lag features)
            features_data = features_data.dropna()
            
            if len(features_data) == 0:
                raise ValueError("No data available after feature creation and NaN removal")
            
            if len(features_data) < 10:
                raise ValueError(f"Insufficient data for CatBoost model (need at least 10 observations, got {len(features_data)})")
            
            X = features_data.drop('sales', axis=1)
            y = features_data['sales']
            
            # Adjust model parameters based on intermittency
            actual_iterations = min(self.iterations, len(features_data) * 15)  # More iterations for sparse data
            
            # Use specialized loss function and parameters for intermittent demand
            if demand_analysis['is_highly_intermittent']:
                # For highly intermittent data (>60% zeros)
                loss_function = 'Poisson'  # Better for count-like sparse data
                learning_rate = max(0.01, self.learning_rate * 0.5)  # Lower learning rate
                depth = 4  # Shallower trees
                l2_reg = 10  # Higher regularization
                self.logger.info("Using Poisson loss for highly intermittent demand")
            elif demand_analysis['is_intermittent']:
                # For moderately intermittent data (25-60% zeros)
                loss_function = 'RMSE'
                learning_rate = max(0.03, self.learning_rate * 0.7)
                depth = 5
                l2_reg = 5
                self.logger.info("Using RMSE loss for intermittent demand")
            else:
                # For regular demand patterns
                loss_function = 'RMSE'
                learning_rate = self.learning_rate
                depth = 6
                l2_reg = 3
                self.logger.info("Using standard parameters for regular demand")
            
            # Initialize CatBoost model with intermittent-demand optimized parameters
            self.model = CatBoostRegressor(
                iterations=actual_iterations,
                learning_rate=learning_rate,
                depth=depth,
                loss_function=loss_function,
                random_state=42,
                verbose=False,
                allow_writing_files=False,
                bootstrap_type='Bernoulli',
                subsample=0.7,  # Lower subsample for sparse data
                colsample_bylevel=0.8,
                reg_lambda=l2_reg,
                random_strength=1,
                border_count=128,  # Fewer borders for sparse data
                min_data_in_leaf=3,  # Smaller leaf requirements
                grow_policy='Lossguide',  # Better for imbalanced data
                max_leaves=32
            )
            
            # Handle class imbalance with sample weights if highly intermittent
            sample_weight = None
            if demand_analysis['is_highly_intermittent']:
                # Give higher weight to non-zero observations
                sample_weight = np.where(y > self.zero_threshold, 3.0, 1.0)
                self.logger.info("Applied sample weights for intermittent demand")
            
            # Fit the model
            if sample_weight is not None:
                self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
            else:
                self.model.fit(X, y, verbose=False)
            
            self.logger.info(f"CatBoost model fitted for intermittent demand with {actual_iterations} iterations on {len(features_data)} records")
            
        except Exception as e:
            self.logger.error(f"Error fitting CatBoost model: {str(e)}")
            raise
    
    def predict(self, train_data: pd.DataFrame, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate predictions with intermittent demand considerations."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            predictions = []
            current_data = train_data.copy()
            
            # Analyze historical pattern for prediction adjustments
            demand_analysis = self.analyze_demand_pattern(train_data['sales'])
            
            # Generate predictions iteratively
            for step in range(steps):
                try:
                    # Create features for the current data
                    features_data = self.create_intermittent_features(current_data)
                    features_data = features_data.dropna()
                    
                    if len(features_data) == 0:
                        # Fallback prediction based on intermittent pattern
                        if predictions:
                            pred = predictions[-1] * 0.9  # Slight decay
                        else:
                            # Use Croston's method as fallback
                            non_zero_sales = train_data['sales'][train_data['sales'] > self.zero_threshold]
                            if len(non_zero_sales) > 0:
                                avg_demand = non_zero_sales.mean()
                                demand_prob = len(non_zero_sales) / len(train_data)
                                pred = avg_demand * demand_prob
                            else:
                                pred = 0
                        predictions.append(max(0, pred))
                        break
                    
                    # Get the last row for prediction
                    X_pred = features_data.drop('sales', axis=1).iloc[-1:].values
                    raw_pred = self.model.predict(X_pred)[0]
                    
                    # Post-processing for intermittent demand
                    if demand_analysis['is_intermittent']:
                        # Apply probability threshold for intermittent demand
                        demand_threshold = self.zero_threshold * 2
                        
                        # If prediction is very low, consider setting to zero
                        if raw_pred < demand_threshold:
                            # Use historical intermittency pattern to decide
                            recent_demand_prob = features_data['demand_prob_7'].iloc[-1] if 'demand_prob_7' in features_data.columns else 0.5
                            
                            # Stochastic element based on historical patterns
                            if recent_demand_prob < 0.2:  # Low recent demand probability
                                pred = 0
                            else:
                                pred = max(0, raw_pred)
                        else:
                            pred = max(0, raw_pred)
                    else:
                        pred = max(0, raw_pred)
                    
                    predictions.append(pred)
                    
                    # Add prediction to data for next iteration
                    next_date = current_data['date'].iloc[-1] + pd.Timedelta(days=1)
                    new_row = pd.DataFrame({
                        'date': [next_date],
                        'sales': [pred]
                    })
                    current_data = pd.concat([current_data, new_row], ignore_index=True)
                    
                    # Keep only recent data to manage memory
                    if len(current_data) > 365:
                        current_data = current_data.iloc[-365:].reset_index(drop=True)
                        
                except Exception as e:
                    self.logger.warning(f"Error in prediction step {step}: {str(e)}")
                    # Fallback prediction
                    if predictions:
                        pred = predictions[-1] * 0.95
                    else:
                        pred = train_data['sales'].mean() if len(train_data) > 0 else 0
                    predictions.append(max(0, pred))
            
            # Create date range for predictions
            start_date = train_data['date'].iloc[-1] + pd.Timedelta(days=1)
            pred_dates = pd.date_range(start=start_date, periods=len(predictions), freq='D')
            
            pred_series = pd.Series(predictions, index=pred_dates)
            
            # Calculate confidence intervals adapted for intermittent demand
            if len(predictions) > 0:
                try:
                    # Calculate residuals from training
                    train_features = self.create_intermittent_features(train_data).dropna()
                    if len(train_features) > 0:
                        X_train = train_features.drop('sales', axis=1)
                        y_train = train_features['sales']
                        train_pred = self.model.predict(X_train)
                        
                        # Separate residuals for zero and non-zero demands
                        zero_mask = y_train <= self.zero_threshold
                        non_zero_residuals = y_train[~zero_mask] - train_pred[~zero_mask] if (~zero_mask).any() else []
                        
                        if len(non_zero_residuals) > 0:
                            residuals_std = np.std(non_zero_residuals)
                        else:
                            residuals_std = pred_series.std() * 0.3  # Conservative estimate
                    else:
                        residuals_std = pred_series.std() * 0.3
                    
                    # Adaptive confidence intervals based on demand level
                    lower_ci = []
                    upper_ci = []
                    
                    for pred in predictions:
                        if pred <= self.zero_threshold:
                            # Narrow intervals for zero/low predictions
                            lower_ci.append(0)
                            upper_ci.append(pred + residuals_std * 0.5)
                        else:
                            # Standard intervals for non-zero predictions
                            lower_ci.append(max(0, pred - 1.96 * residuals_std))
                            upper_ci.append(pred + 1.96 * residuals_std)
                    
                    lower_ci = pd.Series(lower_ci, index=pred_dates)
                    upper_ci = pd.Series(upper_ci, index=pred_dates)
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating confidence intervals: {str(e)}")
                    # Simple fallback intervals
                    lower_ci = pred_series * 0.5
                    upper_ci = pred_series * 1.5
            else:
                lower_ci = pred_series
                upper_ci = pred_series
            
            return pred_series, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in CatBoost prediction: {str(e)}")
            raise
    
    def evaluate(self, train_data: pd.DataFrame, eval_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance with intermittent demand specific metrics."""
        try:
            # Fit on training data
            self.fit(train_data)
            
            # Predict evaluation period
            eval_length = len(eval_data)
            predictions, _, _ = self.predict(train_data, eval_length)
            
            if len(predictions) == 0:
                return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
            
            # Align predictions with evaluation data
            min_length = min(len(predictions), len(eval_data))
            actual = eval_data['sales'].iloc[:min_length].values
            pred = predictions.iloc[:min_length].values
            
            # Standard metrics
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            
            # MAPE with protection against division by zero
            mape_values = []
            for a, p in zip(actual, pred):
                if abs(a) > 1e-8:
                    mape_values.append(abs((a - p) / a))
            mape = np.mean(mape_values) * 100 if mape_values else 0
            
            # R-squared
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Intermittent demand specific metrics
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            # Add intermittent-specific metrics
            zero_threshold = self.zero_threshold
            actual_zeros = (actual <= zero_threshold).sum()
            pred_zeros = (pred <= zero_threshold).sum()
            actual_nonzeros = (actual > zero_threshold).sum()
            pred_nonzeros = (pred > zero_threshold).sum()
            
            # Zero prediction accuracy
            if len(actual) > 0:
                metrics['zero_accuracy'] = np.mean((actual <= zero_threshold) == (pred <= zero_threshold))
            
            # Non-zero MAE (performance on non-zero demands)
            non_zero_mask = actual > zero_threshold
            if non_zero_mask.any():
                metrics['nonzero_MAE'] = mean_absolute_error(actual[non_zero_mask], pred[non_zero_mask])
            
            # Demand occurrence prediction (binary classification accuracy)
            actual_binary = (actual > zero_threshold).astype(int)
            pred_binary = (pred > zero_threshold).astype(int)
            metrics['demand_occurrence_accuracy'] = np.mean(actual_binary == pred_binary)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in CatBoost evaluation: {str(e)}")
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf'), 'R2': 0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including intermittent demand specifics."""
        if self.model is None:
            return {'model_type': 'CatBoost_Intermittent', 'status': 'not_fitted'}
        
        try:
            info = {
                'model_type': 'CatBoost_Intermittent',
                'iterations': self.model.get_param('iterations'),
                'learning_rate': self.model.get_param('learning_rate'),
                'depth': self.model.get_param('depth'),
                'loss_function': self.model.get_param('loss_function'),
                'l2_leaf_reg': self.model.get_param('l2_leaf_reg'),
                'bootstrap_type': self.model.get_param('bootstrap_type'),
                'subsample': self.model.get_param('subsample'),
                'intermittent_ratio': self.intermittent_ratio,
                'zero_threshold': self.zero_threshold,
                'optimization': 'intermittent_demand_specialized'
            }
            
            # Add feature importance if available
            if hasattr(self.model, 'feature_importances_') and self.feature_names:
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                # Get top 10 most important features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                info['top_features'] = dict(top_features)
                
                # Highlight intermittent-specific features
                intermittent_features = [f for f in self.feature_names if any(keyword in f for keyword in 
                                       ['demand_prob', 'time_since', 'croston', 'has_demand', 'consecutive_zeros'])]
                if intermittent_features:
                    intermittent_importance = {f: feature_importance.get(f, 0) for f in intermittent_features}
                    info['intermittent_features'] = dict(sorted(intermittent_importance.items(), 
                                                               key=lambda x: x[1], reverse=True)[:5])
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Error getting model info: {str(e)}")
            return {'model_type': 'CatBoost_Intermittent', 'status': 'error'}
    
    def get_intermittent_insights(self) -> Dict[str, Any]:
        """Get insights specific to intermittent demand patterns."""
        if self.model is None or self.feature_names is None:
            return {}
        
        try:
            insights = {
                'intermittent_ratio': self.intermittent_ratio,
                'model_specialization': 'intermittent_demand',
                'zero_threshold': self.zero_threshold
            }
            
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                
                # Categorize feature importance
                time_features = [f for f in self.feature_names if 'time_since' in f or 'time_until' in f]
                prob_features = [f for f in self.feature_names if 'prob' in f or 'frequency' in f]
                size_features = [f for f in self.feature_names if 'nonzero' in f or 'croston' in f]
                
                insights['feature_categories'] = {
                    'timing_importance': sum(feature_importance.get(f, 0) for f in time_features),
                    'probability_importance': sum(feature_importance.get(f, 0) for f in prob_features),
                    'size_importance': sum(feature_importance.get(f, 0) for f in size_features)
                }
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error getting intermittent insights: {str(e)}")
            return {}