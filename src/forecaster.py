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
        
    def load_data(self, sales_folder_path: str = None) -> None:
        """Load data from specified folder."""
        if sales_folder_path:
            self.data_processor = DataProcessor(sales_folder_path)
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load and prepare all data."""
        try:
            # Combine monthly files
            combined_data = self.data_processor.combine_monthly_files()
            
            # Validate data quality
            cleaned_data = self.data_processor.validate_data_quality(combined_data)
            
            # Get drug names
            drug_names = self.data_processor.get_drug_names(cleaned_data)
            
            # Log data summary
            summary = self.data_processor.get_data_summary(cleaned_data)
            self.logger.info(f"Data loaded successfully:")
            self.logger.info(f"- Records: {summary['total_records']}")
            self.logger.info(f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            self.logger.info(f"- Drugs: {summary['total_drugs']}")
            
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
                self.logger.info(f"Evaluating ARIMA for {drug_name}...")
                arima_metrics = self.arima_model.evaluate(train_data['sales'], eval_data['sales'])
                arima_info = self.arima_model.get_model_info()
                results['ARIMA'] = {
                    'metrics': arima_metrics,
                    'model_info': arima_info,
                    'model': self.arima_model
                }
                self.logger.info(f"ARIMA evaluation complete for {drug_name} - MAE: {arima_metrics.get('MAE', 'N/A'):.2f}")
            except Exception as e:
                self.logger.error(f"ARIMA evaluation failed for {drug_name}: {str(e)}")
                results['ARIMA'] = {'metrics': {}, 'model_info': {}, 'model': None}
            
            # Prophet Model
            try:
                self.logger.info(f"Evaluating Prophet for {drug_name}...")
                prophet_metrics = self.prophet_model.evaluate(train_data, eval_data)
                prophet_info = self.prophet_model.get_model_info()
                results['Prophet'] = {
                    'metrics': prophet_metrics,
                    'model_info': prophet_info,
                    'model': self.prophet_model
                }
                self.logger.info(f"Prophet evaluation complete for {drug_name} - MAE: {prophet_metrics.get('MAE', 'N/A'):.2f}")
            except Exception as e:
                self.logger.error(f"Prophet evaluation failed for {drug_name}: {str(e)}")
                results['Prophet'] = {'metrics': {}, 'model_info': {}, 'model': None}
            
            # CatBoost Model - Enhanced error handling and logging
            try:
                self.logger.info(f"Evaluating CatBoost for {drug_name}...")
                
                # Log data info for debugging
                self.logger.info(f"Train data shape: {train_data.shape}, Eval data shape: {eval_data.shape}")
                self.logger.info(f"Train data columns: {list(train_data.columns)}")
                self.logger.info(f"Sales stats - Train: min={train_data['sales'].min():.2f}, max={train_data['sales'].max():.2f}, mean={train_data['sales'].mean():.2f}")
                
                # Check for zero sales ratio
                zero_ratio = (train_data['sales'] <= 0.01).sum() / len(train_data)
                self.logger.info(f"Zero sales ratio for {drug_name}: {zero_ratio:.1%}")
                
                catboost_metrics = self.catboost_model.evaluate(train_data, eval_data)
                catboost_info = self.catboost_model.get_model_info()
                results['CatBoost'] = {
                    'metrics': catboost_metrics,
                    'model_info': catboost_info,
                    'model': self.catboost_model
                }
                
                # Enhanced logging for CatBoost results
                mae = catboost_metrics.get('MAE', 'N/A')
                zero_acc = catboost_metrics.get('zero_accuracy', 'N/A')
                demand_acc = catboost_metrics.get('demand_occurrence_accuracy', 'N/A')
                
                mae_str = f"{mae:.2f}" if isinstance(mae, (int, float)) and not np.isnan(mae) else "N/A"
                zero_acc_str = f"{zero_acc:.1%}" if isinstance(zero_acc, (int, float)) and not np.isnan(zero_acc) else "N/A"
                demand_acc_str = f"{demand_acc:.1%}" if isinstance(demand_acc, (int, float)) and not np.isnan(demand_acc) else "N/A"
                
                self.logger.info(f"CatBoost evaluation complete for {drug_name} - MAE: {mae_str}, Zero Acc: {zero_acc_str}, Demand Acc: {demand_acc_str}")
                
            except Exception as e:
                self.logger.error(f"CatBoost evaluation failed for {drug_name}: {str(e)}")
                import traceback
                self.logger.error(f"CatBoost error traceback: {traceback.format_exc()}")
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
                self.logger.info(f"Generating ARIMA predictions for {drug_name}...")
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
                self.logger.info(f"ARIMA predictions generated for {drug_name} - {len(arima_pred)} days")
            except Exception as e:
                self.logger.error(f"ARIMA prediction failed for {drug_name}: {str(e)}")
                predictions['ARIMA'] = None
            
            # Prophet Predictions
            try:
                self.logger.info(f"Generating Prophet predictions for {drug_name}...")
                self.prophet_model.fit(drug_data)
                start_date = drug_data['date'].iloc[-1] + pd.Timedelta(days=1)
                prophet_pred, prophet_lower, prophet_upper = self.prophet_model.predict(start_date, self.forecast_days)
                
                predictions['Prophet'] = {
                    'dates': prophet_pred.index,
                    'predictions': prophet_pred.values,
                    'lower_ci': prophet_lower.values,
                    'upper_ci': prophet_upper.values
                }
                self.logger.info(f"Prophet predictions generated for {drug_name} - {len(prophet_pred)} days")
            except Exception as e:
                self.logger.error(f"Prophet prediction failed for {drug_name}: {str(e)}")
                predictions['Prophet'] = None
            
            # CatBoost Predictions - Enhanced error handling
            try:
                self.logger.info(f"Generating CatBoost predictions for {drug_name}...")
                
                # Log input data info
                self.logger.info(f"Input data shape: {drug_data.shape}")
                self.logger.info(f"Date range: {drug_data['date'].min()} to {drug_data['date'].max()}")
                self.logger.info(f"Sales range: {drug_data['sales'].min():.2f} to {drug_data['sales'].max():.2f}")
                
                # Fit the model first
                self.catboost_model.fit(drug_data)
                self.logger.info(f"CatBoost model fitted successfully for {drug_name}")
                
                # Generate predictions
                catboost_pred, catboost_lower, catboost_upper = self.catboost_model.predict(drug_data, self.forecast_days)
                
                # Validate predictions
                if len(catboost_pred) == 0:
                    raise ValueError("CatBoost returned empty predictions")
                
                predictions['CatBoost'] = {
                    'dates': catboost_pred.index,
                    'predictions': catboost_pred.values,
                    'lower_ci': catboost_lower.values,
                    'upper_ci': catboost_upper.values
                }
                
                # Log prediction summary
                pred_mean = np.mean(catboost_pred.values)
                pred_std = np.std(catboost_pred.values)
                zero_count = (catboost_pred.values <= 0.01).sum()
                
                self.logger.info(f"CatBoost predictions generated for {drug_name} - {len(catboost_pred)} days")
                self.logger.info(f"Prediction stats - Mean: {pred_mean:.2f}, Std: {pred_std:.2f}, Zeros: {zero_count}/{len(catboost_pred)}")
                
            except Exception as e:
                self.logger.error(f"CatBoost prediction failed for {drug_name}: {str(e)}")
                import traceback
                self.logger.error(f"CatBoost prediction error traceback: {traceback.format_exc()}")
                predictions['CatBoost'] = None
            
            # Log overall prediction results
            successful_models = [model for model, pred in predictions.items() if pred is not None]
            self.logger.info(f"Successful predictions for {drug_name}: {successful_models}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {drug_name}: {str(e)}")
            return {}
    
    def create_visualization(self, drug_data: pd.DataFrame, predictions: Dict, 
                           evaluation_results: Dict, drug_name: str, model_name: str) -> plt.Figure:
        """Create visualization for predictions."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Historical data and predictions
            ax1.plot(drug_data['date'], drug_data['sales'], label='Historical Sales', color='blue', linewidth=2)
            
            if predictions.get(model_name) is not None:
                pred_data = predictions[model_name]
                ax1.plot(pred_data['dates'], pred_data['predictions'], 
                        label=f'{model_name} Forecast', color='red', linewidth=2)
                ax1.fill_between(pred_data['dates'], pred_data['lower_ci'], pred_data['upper_ci'],
                               alpha=0.3, color='red', label='Confidence Interval')
            else:
                ax1.text(0.5, 0.5, f'No predictions available for {model_name}', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            
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
                
                # Filter out infinite or NaN values
                valid_metrics = []
                valid_names = []
                for name, value in zip(metric_names, metric_values):
                    if not (np.isnan(value) or np.isinf(value)):
                        valid_metrics.append(value)
                        valid_names.append(name)
                
                if valid_metrics:
                    bars = ax2.bar(valid_names, valid_metrics, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
                    ax2.set_title(f'{model_name} Model Performance Metrics', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Value')
                    
                    # Add value labels on bars
                    for bar, value, name in zip(bars, valid_metrics, valid_names):
                        if 'accuracy' in name.lower():
                            label = f'{value:.1%}'
                        elif name == 'MAPE':
                            label = f'{value:.1f}%'
                        else:
                            label = f'{value:.2f}'
                        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(valid_metrics)*0.01,
                               label, ha='center', va='bottom', fontsize=9)
                    
                    # Rotate x-axis labels if needed
                    ax2.tick_params(axis='x', rotation=45)
                else:
                    ax2.text(0.5, 0.5, 'No valid evaluation metrics available', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            else:
                ax2.text(0.5, 0.5, 'No evaluation metrics available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            
            ax2.set_title(f'{model_name} Model Performance Metrics', fontsize=12, fontweight='bold')
            
            # Plot 3: Sales distribution
            ax3.hist(drug_data['sales'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(drug_data['sales'].mean(), color='red', linestyle='--', label=f'Mean: {drug_data["sales"].mean():.2f}')
            ax3.axvline(drug_data['sales'].median(), color='green', linestyle='--', label=f'Median: {drug_data["sales"].median():.2f}')
            ax3.set_title(f'{drug_name} Sales Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Sales')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Monthly sales trend
            try:
                drug_data_monthly = drug_data.copy()
                drug_data_monthly['month'] = drug_data_monthly['date'].dt.to_period('M')
                monthly_sales = drug_data_monthly.groupby('month')['sales'].mean()
                
                ax4.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', linewidth=2, markersize=4)
                ax4.set_title(f'{drug_name} Monthly Average Sales', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Month')
                ax4.set_ylabel('Average Sales')
                ax4.grid(True, alpha=0.3)
                ax4.tick_params(axis='x', rotation=45)
                
                # Add trend line
                x_numeric = range(len(monthly_sales))
                z = np.polyfit(x_numeric, monthly_sales.values, 1)
                p = np.poly1d(z)
                ax4.plot(monthly_sales.index.astype(str), p(x_numeric), "--", alpha=0.8, color='red', label='Trend')
                ax4.legend()
                
            except Exception as e:
                ax4.text(0.5, 0.5, f'Error creating monthly trend:\n{str(e)}', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            
            plt.suptitle(f'{drug_name} - {model_name} Analysis Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            # Return a simple figure with error message
            fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
            ax.text(0.5, 0.5, f'Error creating visualization:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{drug_name} - {model_name} (Error)')
            return fig
    
    def save_outputs(self, drug_name: str, predictions: Dict, evaluation_results: Dict, 
                    drug_data: pd.DataFrame, latest_month: str) -> None:
        """Save all outputs to files following the instruction format."""
        try:
            # Create output directory structure as per instructions
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
                    
                    # Follow instruction naming: drugname_modelname_MM_YYYY.csv
                    csv_filename = f"{drug_name}_{model_name.lower()}_{latest_month}.csv"
                    csv_path = os.path.join(model_path, csv_filename)
                    pred_df.to_csv(csv_path, index=False)
                    self.logger.info(f"Saved predictions to {csv_path}")
                else:
                    self.logger.warning(f"No predictions available for {drug_name} - {model_name}")
                
                # Save visualization
                fig = self.create_visualization(drug_data, predictions, evaluation_results, drug_name, model_name)
                png_filename = f"{drug_name}_{model_name.lower()}_{latest_month}.png"
                png_path = os.path.join(model_path, png_filename)
                fig.savefig(png_path, dpi=config.DPI, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"Saved visualization to {png_path}")
                
                # Save evaluation metrics
                txt_filename = f"{drug_name}_{model_name.lower()}_{latest_month}.txt"
                txt_path = os.path.join(model_path, txt_filename)
                
                with open(txt_path, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Drug: {drug_name}\n")
                    f.write(f"Training Period: {drug_data['date'].iloc[0].strftime('%Y-%m-%d')} to {drug_data['date'].iloc[-1].strftime('%Y-%m-%d')}\n")
                    f.write(f"Forecast Period: {self.forecast_days} days\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Add prediction status
                    if predictions.get(model_name) is not None:
                        f.write(f"Prediction Status: SUCCESS\n")
                        f.write(f"Predictions Generated: {len(predictions[model_name]['predictions'])}\n\n")
                    else:
                        f.write(f"Prediction Status: FAILED\n\n")
                    
                    # Add intermittent demand analysis for CatBoost
                    if model_name == 'CatBoost' and model_name in evaluation_results:
                        model_info = evaluation_results[model_name].get('model_info', {})
                        if 'intermittent_ratio' in model_info:
                            f.write("Demand Pattern Analysis:\n")
                            f.write(f"- Intermittent ratio: {model_info['intermittent_ratio']:.1%} zeros\n")
                            if model_info['intermittent_ratio'] > config.HIGHLY_INTERMITTENT_THRESHOLD:
                                f.write("- Classification: Highly intermittent demand\n")
                            elif model_info['intermittent_ratio'] > config.INTERMITTENT_RATIO_THRESHOLD:
                                f.write("- Classification: Intermittent demand\n")
                            else:
                                f.write("- Classification: Regular demand\n")
                            f.write("\n")
                    
                    if model_name in evaluation_results and evaluation_results[model_name]['metrics']:
                        metrics = evaluation_results[model_name]['metrics']
                        f.write("Evaluation Metrics:\n")
                        for metric, value in metrics.items():
                            if not np.isnan(value) and not np.isinf(value):
                                if metric == 'MAPE':
                                    f.write(f"- {metric}: {value:.2f}%\n")
                                elif 'accuracy' in metric.lower():
                                    f.write(f"- {metric}: {value:.1%}\n")
                                else:
                                    f.write(f"- {metric}: {value:.2f}\n")
                            else:
                                f.write(f"- {metric}: N/A\n")
                    else:
                        f.write("Evaluation Metrics: Not available\n")
                    
                    if model_name in evaluation_results and evaluation_results[model_name]['model_info']:
                        model_info = evaluation_results[model_name]['model_info']
                        f.write(f"\nModel Parameters:\n")
                        for param, value in model_info.items():
                            if isinstance(value, dict):
                                f.write(f"- {param}: {len(value)} items\n")
                            else:
                                f.write(f"- {param}: {value}\n")
                
                self.logger.info(f"Saved evaluation metrics to {txt_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving outputs: {str(e)}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get stored results from the last analysis."""
        return getattr(self, '_last_results', {})
    
    def run_complete_analysis(self, drug_filter: List[str] = None) -> Dict[str, Any]:
        """Run complete analysis for all drugs or filtered drugs."""
        try:
            # Load and prepare data
            combined_data, drug_names = self.load_and_prepare_data()
            
            # Filter drugs if specified
            if drug_filter:
                available_drugs = set(drug_names)
                filtered_drugs = [drug for drug in drug_filter if drug in available_drugs]
                missing_drugs = set(drug_filter) - available_drugs
                
                if missing_drugs:
                    self.logger.warning(f"Drugs not found in dataset: {missing_drugs}")
                
                if not filtered_drugs:
                    raise ValueError(f"None of the specified drugs found in dataset. Available: {drug_names}")
                    
                drug_names = filtered_drugs
                self.logger.info(f"Filtered to {len(drug_names)} drugs: {drug_names}")
            
            # Get latest month for output folder naming
            latest_month = self.data_processor.get_latest_month()
            
            results = {}
            successful_drugs = 0
            
            for i, drug_name in enumerate(drug_names, 1):
                self.logger.info(f"Processing {drug_name} ({i}/{len(drug_names)})...")
                
                try:
                    # Prepare drug-specific data
                    drug_data = self.data_processor.prepare_time_series(combined_data, drug_name)
                    
                    if len(drug_data) < 30:  # Minimum data requirement
                        self.logger.warning(f"Insufficient data for {drug_name} (only {len(drug_data)} records)")
                        continue
                    
                    # Log drug data summary
                    zero_ratio = (drug_data['sales'] <= 0.01).sum() / len(drug_data)
                    self.logger.info(f"{drug_name} - Data points: {len(drug_data)}, Zero ratio: {zero_ratio:.1%}, "
                                   f"Mean sales: {drug_data['sales'].mean():.2f}")
                    
                    # Train and evaluate models
                    evaluation_results = self.train_evaluate_models(drug_data, drug_name)
                    
                    # Generate final predictions
                    predictions = self.generate_predictions(drug_data, drug_name)
                    
                    # Save outputs
                    self.save_outputs(drug_name, predictions, evaluation_results, drug_data, latest_month)
                    
                    results[drug_name] = {
                        'evaluation_results': evaluation_results,
                        'predictions': predictions,
                        'data_points': len(drug_data),
                        'zero_ratio': zero_ratio
                    }
                    
                    successful_drugs += 1
                    self.logger.info(f"Completed analysis for {drug_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {drug_name}: {str(e)}")
                    import traceback
                    self.logger.error(f"Error traceback: {traceback.format_exc()}")
                    continue
            
            # Store results for get_results() method
            self._last_results = results
            
            self.logger.info(f"Analysis complete. Successfully processed {successful_drugs}/{len(drug_names)} drugs.")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {str(e)}")
            raise