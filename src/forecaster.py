import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from data_processor import DataProcessor
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.catboost_model import CatBoostModel
import config

# Set plot style
plt.style.use(config.PLOT_STYLE)
sns.set_palette("husl")

class Forecaster:
    def __init__(self, forecast_days: int = config.FORECAST_DAYS):
        self.forecast_days = forecast_days
        self.data_processor = DataProcessor(config.DATA_FOLDER)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.arima_model = ARIMAModel(seasonal_periods=config.ARIMA_SEASONAL_PERIODS)
        self.prophet_model = ProphetModel(seasonality_mode=config.PROPHET_SEASONALITY_MODE)
        self.catboost_model = CatBoostModel(
            iterations=config.CATBOOST_ITERATIONS,
            learning_rate=config.CATBOOST_LEARNING_RATE
        )
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load and prepare all data."""
        try:
            # Combine monthly files
            combined_data = self.data_processor.combine_monthly_files()
            
            # Validate data quality
            cleaned_data = self.data_processor.validate_data_quality(combined_data)
            
            # Get drug names
            drug_names = self.data_processor.get_drug_names(cleaned_data)
            
            self.logger.info(f"Data prepared successfully. Found {len(drug_names)} drugs")
            
            return cleaned_data, drug_names
            
        except Exception as e:
            self.logger.error(f"Error loading and preparing data: {str(e)}")
            raise
    
    def train_evaluate_models(self, drug_data: pd.DataFrame, drug_name: str) -> Dict[str, Dict]:
        """Train and evaluate all models for a specific drug."""
        try:
            # Split data
            train_data, eval_data = self.data_processor.split_data(
                drug_data, train_ratio=config.TRAIN_TEST_SPLIT
            )
            
            results = {}
            
            # ARIMA Model
            try:
                arima_metrics = self.arima_model.evaluate(train_data['sales'], eval_data['sales'])
                arima_info = self.arima_model.get_model_info()
                results['ARIMA'] = {
                    'metrics': arima_metrics,
                    'model_info': arima_info,
                    'model': self.arima_model
                }
                self.logger.info(f"ARIMA evaluation complete for {drug_name}")
            except Exception as e:
                self.logger.error(f"ARIMA evaluation failed for {drug_name}: {str(e)}")
                results['ARIMA'] = {'metrics': {}, 'model_info': {}, 'model': None}
            
            # Prophet Model
            try:
                prophet_metrics = self.prophet_model.evaluate(train_data, eval_data)
                prophet_info = self.prophet_model.get_model_info()
                results['Prophet'] = {
                    'metrics': prophet_metrics,
                    'model_info': prophet_info,
                    'model': self.prophet_model
                }
                self.logger.info(f"Prophet evaluation complete for {drug_name}")
            except Exception as e:
                self.logger.error(f"Prophet evaluation failed for {drug_name}: {str(e)}")
                results['Prophet'] = {'metrics': {}, 'model_info': {}, 'model': None}
            
            # CatBoost Model
            try:
                catboost_metrics = self.catboost_model.evaluate(train_data, eval_data)
                catboost_info = self.catboost_model.get_model_info()
                results['CatBoost'] = {
                    'metrics': catboost_metrics,
                    'model_info': catboost_info,
                    'model': self.catboost_model
                }
                self.logger.info(f"CatBoost evaluation complete for {drug_name}")
            except Exception as e:
                self.logger.error(f"CatBoost evaluation failed for {drug_name}: {str(e)}")
                results['CatBoost'] = {'metrics': {}, 'model_info': {}, 'model': None}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model training/evaluation for {drug_name}: {str(e)}")
            return {}
    
    def generate_predictions(self, drug_data: pd.DataFrame, drug_name: str) -> Dict[str, Dict]:
        """Generate final predictions using full dataset."""
        try:
            predictions = {}
            
            # ARIMA Predictions
            try:
                self.arima_model.fit(drug_data['sales'])
                arima_pred, arima_lower, arima_upper = self.arima_model.predict(self.forecast_days)
                
                start_date = drug_data['date'].iloc[-1] + pd.Timedelta(days=1)
                pred_dates = pd.date_range(start=start_date, periods=self.forecast_days, freq='D')
                
                predictions['ARIMA'] = {
                    'dates': pred_dates,
                    'predictions': arima_pred.values,
                    'lower_ci': arima_lower.values,
                    'upper_ci': arima_upper.values
                }
                self.logger.info(f"ARIMA predictions generated for {drug_name}")
            except Exception as e:
                self.logger.error(f"ARIMA prediction failed for {drug_name}: {str(e)}")
                predictions['ARIMA'] = None
            
            # Prophet Predictions
            try:
                self.prophet_model.fit(drug_data)
                start_date = drug_data['date'].iloc[-1] + pd.Timedelta(days=1)
                prophet_pred, prophet_lower, prophet_upper = self.prophet_model.predict(start_date, self.forecast_days)
                
                predictions['Prophet'] = {
                    'dates': prophet_pred.index,
                    'predictions': prophet_pred.values,
                    'lower_ci': prophet_lower.values,
                    'upper_ci': prophet_upper.values
                }
                self.logger.info(f"Prophet predictions generated for {drug_name}")
            except Exception as e:
                self.logger.error(f"Prophet prediction failed for {drug_name}: {str(e)}")
                predictions['Prophet'] = None
            
            # CatBoost Predictions
            try:
                self.catboost_model.fit(drug_data)
                catboost_pred, catboost_lower, catboost_upper = self.catboost_model.predict(drug_data, self.forecast_days)
                
                predictions['CatBoost'] = {
                    'dates': catboost_pred.index,
                    'predictions': catboost_pred.values,
                    'lower_ci': catboost_lower.values,
                    'upper_ci': catboost_upper.values
                }
                self.logger.info(f"CatBoost predictions generated for {drug_name}")
            except Exception as e:
                self.logger.error(f"CatBoost prediction failed for {drug_name}: {str(e)}")
                predictions['CatBoost'] = None
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {drug_name}: {str(e)}")
            return {}
    
    def create_visualization(self, drug_data: pd.DataFrame, predictions: Dict, 
                           evaluation_results: Dict, drug_name: str, model_name: str) -> plt.Figure:
        """Create visualization for predictions."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.FIGURE_SIZE)
            
            # Plot 1: Historical data and predictions
            ax1.plot(drug_data['date'], drug_data['sales'], label='Historical Sales', color='blue', linewidth=2)
            
            if predictions[model_name] is not None:
                pred_data = predictions[model_name]
                ax1.plot(pred_data['dates'], pred_data['predictions'], 
                        label=f'{model_name} Forecast', color='red', linewidth=2)
                ax1.fill_between(pred_data['dates'], pred_data['lower_ci'], pred_data['upper_ci'],
                               alpha=0.3, color='red', label='Confidence Interval')
            
            ax1.set_title(f'{drug_name} - {model_name} Sales Forecast', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Sales')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Model performance metrics
            if model_name in evaluation_results and evaluation_results[model_name]['metrics']:
                metrics = evaluation_results[model_name]['metrics']
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                bars = ax2.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
                ax2.set_title(f'{model_name} Model Performance Metrics', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Value')
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    if not np.isnan(value) and not np.isinf(value):
                        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(metric_values)*0.01,
                               f'{value:.2f}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No evaluation metrics available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title(f'{model_name} Model Performance Metrics', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            # Return a simple figure with error message
            fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
            ax.text(0.5, 0.5, f'Error creating visualization:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def save_outputs(self, drug_name: str, predictions: Dict, evaluation_results: Dict, 
                    drug_data: pd.DataFrame, latest_month: str) -> None:
        """Save all outputs to files."""
        try:
            # Create output directory structure
            base_path = os.path.join(config.OUTPUT_BASE_PATH, f'sales_{latest_month}')
            os.makedirs(base_path, exist_ok=True)
            
            for model_name in ['ARIMA', 'Prophet', 'CatBoost']:
                model_path = os.path.join(base_path, model_name)
                os.makedirs(model_path, exist_ok=True)
                
                # Save prediction CSV
                if predictions.get(model_name) is not None:
                    pred_data = predictions[model_name]
                    pred_df = pd.DataFrame({
                        'date': pred_data['dates'],
                        'drug_name': drug_name,
                        'predicted_sales': pred_data['predictions'],
                        'confidence_lower': pred_data['lower_ci'],
                        'confidence_upper': pred_data['upper_ci']
                    })
                    
                    csv_filename = f"{model_name.lower()}_{latest_month}.csv"
                    csv_path = os.path.join(model_path, csv_filename)
                    pred_df.to_csv(csv_path, index=False)
                    self.logger.info(f"Saved predictions to {csv_path}")
                
                # Save visualization
                fig = self.create_visualization(drug_data, predictions, evaluation_results, drug_name, model_name)
                png_filename = f"{model_name.lower()}_{latest_month}.png"
                png_path = os.path.join(model_path, png_filename)
                fig.savefig(png_path, dpi=config.DPI, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"Saved visualization to {png_path}")
                
                # Save evaluation metrics
                txt_filename = f"{model_name.lower()}_{latest_month}.txt"
                txt_path = os.path.join(model_path, txt_filename)
                
                with open(txt_path, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Drug: {drug_name}\n")
                    f.write(f"Training Period: {drug_data['date'].iloc[0].strftime('%Y-%m-%d')} to {drug_data['date'].iloc[-1].strftime('%Y-%m-%d')}\n")
                    f.write(f"Forecast Period: {self.forecast_days} days\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    if model_name in evaluation_results and evaluation_results[model_name]['metrics']:
                        metrics = evaluation_results[model_name]['metrics']
                        f.write("Evaluation Metrics:\n")
                        for metric, value in metrics.items():
                            if not np.isnan(value) and not np.isinf(value):
                                if metric == 'MAPE':
                                    f.write(f"- {metric}: {value:.2f}%\n")
                                else:
                                    f.write(f"- {metric}: {value:.2f}\n")
                            else:
                                f.write(f"- {metric}: N/A\n")
                    
                    if model_name in evaluation_results and evaluation_results[model_name]['model_info']:
                        model_info = evaluation_results[model_name]['model_info']
                        f.write(f"\nModel Parameters:\n")
                        for param, value in model_info.items():
                            f.write(f"- {param}: {value}\n")
                
                self.logger.info(f"Saved evaluation metrics to {txt_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving outputs: {str(e)}")
    
    def run_complete_analysis(self, drug_filter: List[str] = None) -> Dict[str, Any]:
        """Run complete analysis for all drugs or filtered drugs."""
        try:
            # Load and prepare data
            combined_data, drug_names = self.load_and_prepare_data()
            
            # Filter drugs if specified
            if drug_filter:
                drug_names = [drug for drug in drug_names if drug in drug_filter]
                self.logger.info(f"Filtered to {len(drug_names)} drugs: {drug_names}")
            
            # Get latest month for output folder naming
            latest_month = self.data_processor.get_latest_month()
            
            results = {}
            
            for drug_name in drug_names:
                self.logger.info(f"Processing {drug_name}...")
                
                try:
                    # Prepare drug-specific data
                    drug_data = self.data_processor.prepare_time_series(combined_data, drug_name)
                    
                    if len(drug_data) < 30:  # Minimum data requirement
                        self.logger.warning(f"Insufficient data for {drug_name} (only {len(drug_data)} records)")
                        continue
                    
                    # Train and evaluate models
                    evaluation_results = self.train_evaluate_models(drug_data, drug_name)
                    
                    # Generate final predictions
                    predictions = self.generate_predictions(drug_data, drug_name)
                    
                    # Save outputs
                    self.save_outputs(drug_name, predictions, evaluation_results, drug_data, latest_month)
                    
                    results[drug_name] = {
                        'evaluation_results': evaluation_results,
                        'predictions': predictions,
                        'data_points': len(drug_data)
                    }
                    
                    self.logger.info(f"Completed analysis for {drug_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {drug_name}: {str(e)}")
                    continue
            
            self.logger.info(f"Analysis complete. Processed {len(results)} drugs successfully.")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {str(e)}")
            raise