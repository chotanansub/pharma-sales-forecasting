#!/usr/bin/env python3
"""
Pharmacy Sales Prediction Module - Main Entry Point
"""

import argparse
import logging
import sys
import os
from typing import List, Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecaster import Forecaster
import config

def setup_logging(log_level: str = config.LOG_LEVEL) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pharmacy_prediction.log')
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pharmacy Sales Prediction Module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --forecast_days 365               # Custom forecast period
  python main.py --drug_filter M01AB,N02BA         # Specific drugs only
  python main.py --output_path custom_predictions/ # Custom output location
  python main.py --data_path custom_data/sales/    # Custom data location
        """
    )
    
    parser.add_argument(
        '--forecast_days',
        type=int,
        default=config.FORECAST_DAYS,
        help=f'Number of days to forecast (default: {config.FORECAST_DAYS})'
    )
    
    parser.add_argument(
        '--drug_filter',
        type=str,
        help='Comma-separated list of specific drugs to analyze (e.g., M01AB,N02BA,N05B)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=config.OUTPUT_BASE_PATH,
        help=f'Output directory path (default: {config.OUTPUT_BASE_PATH})'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=config.DATA_FOLDER,
        help=f'Input data directory path (default: {config.DATA_FOLDER})'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default=config.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'Logging level (default: {config.LOG_LEVEL})'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=config.TRAIN_TEST_SPLIT,
        help=f'Training data ratio for evaluation (default: {config.TRAIN_TEST_SPLIT})'
    )
    
    return parser.parse_args()

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.forecast_days <= 0:
        raise ValueError("Forecast days must be positive")
    
    if not 0.1 <= args.train_ratio <= 0.9:
        raise ValueError("Train ratio must be between 0.1 and 0.9")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

def main() -> None:
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("="*60)
        logger.info("PHARMACY SALES PREDICTION MODULE")
        logger.info("="*60)
        logger.info(f"Forecast days: {args.forecast_days}")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Output path: {args.output_path}")
        logger.info(f"Train ratio: {args.train_ratio}")
        
        # Update config with command line arguments
        config.FORECAST_DAYS = args.forecast_days
        config.OUTPUT_BASE_PATH = args.output_path
        config.DATA_FOLDER = args.data_path
        config.TRAIN_TEST_SPLIT = args.train_ratio
        
        # Parse drug filter
        drug_filter = None
        if args.drug_filter:
            drug_filter = [drug.strip() for drug in args.drug_filter.split(',')]
            logger.info(f"Drug filter: {drug_filter}")
        
        # Initialize forecaster
        forecaster = Forecaster(forecast_days=args.forecast_days)
        
        # Run complete analysis
        logger.info("Starting analysis...")
        results = forecaster.run_complete_analysis(drug_filter=drug_filter)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        if not results:
            logger.warning("No drugs were successfully processed!")
            return
        
        # Summary statistics
        total_drugs = len(results)
        avg_data_points = sum(drug['data_points'] for drug in results.values()) / total_drugs
        
        logger.info(f"Successfully processed drugs: {total_drugs}")
        logger.info(f"Average data points per drug: {avg_data_points:.1f}")
        
        # Intermittent demand analysis
        intermittent_drugs = 0
        highly_intermittent_drugs = 0
        
        # Model performance summary
        model_success_rates = {'ARIMA': 0, 'Prophet': 0, 'CatBoost': 0}
        model_avg_metrics = {'ARIMA': {}, 'Prophet': {}, 'CatBoost': {}}
        intermittent_performance = {}
        
        for drug_name, drug_results in results.items():
            logger.info(f"\nDrug: {drug_name}")
            logger.info(f"Data points: {drug_results['data_points']}")
            
            # Check for intermittent demand information
            catboost_info = drug_results['evaluation_results'].get('CatBoost', {}).get('model_info', {})
            if 'intermittent_ratio' in catboost_info:
                intermittent_ratio = catboost_info['intermittent_ratio']
                if intermittent_ratio > config.HIGHLY_INTERMITTENT_THRESHOLD:
                    highly_intermittent_drugs += 1
                    logger.info(f"  Highly intermittent demand: {intermittent_ratio:.1%} zeros")
                elif intermittent_ratio > config.INTERMITTENT_RATIO_THRESHOLD:
                    intermittent_drugs += 1
                    logger.info(f"  Intermittent demand: {intermittent_ratio:.1%} zeros")
            
            for model_name, model_results in drug_results['evaluation_results'].items():
                if model_results['metrics']:
                    model_success_rates[model_name] += 1
                    metrics = model_results['metrics']
                    
                    # Collect metrics for averaging
                    for metric, value in metrics.items():
                        if not (np.isnan(value) or np.isinf(value)):
                            if metric not in model_avg_metrics[model_name]:
                                model_avg_metrics[model_name][metric] = []
                            model_avg_metrics[model_name][metric].append(value)
                    
                    # Special reporting for intermittent-specific metrics
                    if model_name == 'CatBoost' and 'zero_accuracy' in metrics:
                        logger.info(f"  {model_name} - MAE: {metrics.get('MAE', 'N/A'):.2f}, "
                                  f"RMSE: {metrics.get('RMSE', 'N/A'):.2f}, "
                                  f"Zero Acc: {metrics.get('zero_accuracy', 'N/A'):.1%}, "
                                  f"Demand Occ Acc: {metrics.get('demand_occurrence_accuracy', 'N/A'):.1%}")
                        
                        # Collect intermittent performance
                        if 'intermittent_ratio' in catboost_info:
                            if catboost_info['intermittent_ratio'] not in intermittent_performance:
                                intermittent_performance[catboost_info['intermittent_ratio']] = []
                            intermittent_performance[catboost_info['intermittent_ratio']].append(metrics)
                    else:
                        logger.info(f"  {model_name} - MAE: {metrics.get('MAE', 'N/A'):.2f}, "
                                  f"RMSE: {metrics.get('RMSE', 'N/A'):.2f}, "
                                  f"MAPE: {metrics.get('MAPE', 'N/A'):.2f}%, "
                                  f"R²: {metrics.get('R2', 'N/A'):.3f}")
        
        # Intermittent demand summary
        if intermittent_drugs + highly_intermittent_drugs > 0:
            logger.info(f"\n" + "="*40)
            logger.info("INTERMITTENT DEMAND ANALYSIS")
            logger.info("="*40)
            logger.info(f"Regular demand drugs: {total_drugs - intermittent_drugs - highly_intermittent_drugs}")
            logger.info(f"Intermittent demand drugs: {intermittent_drugs}")
            logger.info(f"Highly intermittent demand drugs: {highly_intermittent_drugs}")
            logger.info(f"Total intermittent ratio: {(intermittent_drugs + highly_intermittent_drugs)/total_drugs:.1%}")
        
        # Overall model performance
        
        for drug_name, drug_results in results.items():
            logger.info(f"\nDrug: {drug_name}")
            logger.info(f"Data points: {drug_results['data_points']}")
            
            for model_name, model_results in drug_results['evaluation_results'].items():
                if model_results['metrics']:
                    model_success_rates[model_name] += 1
                    metrics = model_results['metrics']
                    
                    # Collect metrics for averaging
                    for metric, value in metrics.items():
                        if not (np.isnan(value) or np.isinf(value)):
                            if metric not in model_avg_metrics[model_name]:
                                model_avg_metrics[model_name][metric] = []
                            model_avg_metrics[model_name][metric].append(value)
                    
                    logger.info(f"  {model_name} - MAE: {metrics.get('MAE', 'N/A'):.2f}, "
                              f"RMSE: {metrics.get('RMSE', 'N/A'):.2f}, "
                              f"MAPE: {metrics.get('MAPE', 'N/A'):.2f}%, "
                              f"R²: {metrics.get('R2', 'N/A'):.3f}")
        
        # Overall model performance
        logger.info(f"\n" + "="*40)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*40)
        
        for model_name in ['ARIMA', 'Prophet', 'CatBoost']:
            success_rate = (model_success_rates[model_name] / total_drugs) * 100
            logger.info(f"\n{model_name}:")
            logger.info(f"  Success rate: {success_rate:.1f}% ({model_success_rates[model_name]}/{total_drugs})")
            
            if model_avg_metrics[model_name]:
                logger.info(f"  Average metrics:")
                for metric, values in model_avg_metrics[model_name].items():
                    if values:
                        avg_value = sum(values) / len(values)
                        if metric == 'MAPE':
                            logger.info(f"    {metric}: {avg_value:.2f}%")
                        else:
                            logger.info(f"    {metric}: {avg_value:.3f}")
        
        logger.info(f"\n" + "="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {config.OUTPUT_BASE_PATH}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Analysis failed: {str(e)}")
        logger.error("Please check the error above and verify your data files and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()