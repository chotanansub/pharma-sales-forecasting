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
  python main.py --drug_filter aspirin,ibuprofen   # Specific drugs only
  python main.py --output_path custom_predictions/  # Custom output location
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
        help='Comma-separated list of specific drugs to analyze'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=config.OUTPUT_BASE_PATH,
        help=f'Output directory path (default: {config.OUTPUT_BASE_PATH})'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default=config.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'Logging level (default: {config.LOG_LEVEL})'
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Pharmacy Sales Prediction Module")
        logger.info(f"Forecast days: {args.forecast_days}")
        logger.info(f"Output path: {args.output_path}")
        
        # Update config with command line arguments
        config.FORECAST_DAYS = args.forecast_days
        config.OUTPUT_BASE_PATH = args.output_path
        
        # Parse drug filter
        drug_filter = None
        if args.drug_filter:
            drug_filter = [drug.strip() for drug in args.drug_filter.split(',')]
            logger.info(f"Drug filter: {drug_filter}")
        
        # Initialize forecaster
        forecaster = Forecaster(forecast_days=args.forecast_days)
        
        # Run complete analysis
        results = forecaster.run_complete_analysis(drug_filter=drug_filter)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*50)
        
        for drug_name, drug_results in results.items():
            logger.info(f"\nDrug: {drug_name}")
            logger.info(f"Data points: {drug_results['data_points']}")
            
            for model_name, model_results in drug_results['evaluation_results'].items():
                if model_results['metrics']:
                    metrics = model_results['metrics']
                    logger.info(f"{model_name} - MAE: {metrics.get('MAE', 'N/A'):.2f}, "
                              f"RMSE: {metrics.get('RMSE', 'N/A'):.2f}, "
                              f"MAPE: {metrics.get('MAPE', 'N/A'):.2f}%, "
                              f"R²: {metrics.get('R2', 'N/A'):.3f}")
        
        logger.info(f"\nAnalysis completed successfully!")
        logger.info(f"Results saved to: {config.OUTPUT_BASE_PATH}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()