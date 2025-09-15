import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional

class DataProcessor:
    def __init__(self, sales_folder_path: str):
        self.sales_folder_path = sales_folder_path
        self.logger = logging.getLogger(__name__)
        
    def combine_monthly_files(self) -> pd.DataFrame:
        """Combine all monthly sales files into a single dataset."""
        try:
            all_files = []
            # Fixed pattern to match instruction format: sales_YYYY_MM.csv
            file_pattern = r'sales_(\d{4})_(\d{2})\.csv'
            
            # Get all CSV files in the sales folder
            if not os.path.exists(self.sales_folder_path):
                raise FileNotFoundError(f"Sales folder not found: {self.sales_folder_path}")
                
            files = [f for f in os.listdir(self.sales_folder_path) if f.endswith('.csv')]
            
            if not files:
                raise FileNotFoundError(f"No CSV files found in {self.sales_folder_path}")
            
            # Sort files by date
            dated_files = []
            for file in files:
                match = re.match(file_pattern, file)
                if match:
                    year, month = match.groups()
                    dated_files.append((int(year), int(month), file))
                else:
                    self.logger.warning(f"File {file} doesn't match expected pattern sales_YYYY_MM.csv")
            
            if not dated_files:
                raise ValueError("No files found matching the expected naming pattern sales_YYYY_MM.csv")
            
            dated_files.sort()
            
            # Read and combine files
            for year, month, file in dated_files:
                file_path = os.path.join(self.sales_folder_path, file)
                self.logger.info(f"Reading file: {file}")
                
                df = pd.read_csv(file_path)
                df['source_file'] = file
                all_files.append(df)
            
            # Combine all dataframes
            combined_data = pd.concat(all_files, ignore_index=True)
            self.logger.info(f"Combined {len(all_files)} files into dataset with {len(combined_data)} rows")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error combining monthly files: {str(e)}")
            raise
    
    def validate_data_quality(self, combined_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the combined data."""
        try:
            # Convert date column to datetime
            combined_data['date'] = pd.to_datetime(combined_data['date'])
            
            # Sort by date
            combined_data = combined_data.sort_values('date').reset_index(drop=True)
            
            # Remove duplicates based on date
            initial_rows = len(combined_data)
            combined_data = combined_data.drop_duplicates(subset='date', keep='first')
            removed_duplicates = initial_rows - len(combined_data)
            
            if removed_duplicates > 0:
                self.logger.warning(f"Removed {removed_duplicates} duplicate date entries")
            
            # Check for missing dates
            date_range = pd.date_range(start=combined_data['date'].min(), 
                                     end=combined_data['date'].max(), 
                                     freq='D')
            missing_dates = set(date_range) - set(combined_data['date'])
            
            if missing_dates:
                self.logger.warning(f"Found {len(missing_dates)} missing dates in the dataset")
                # Fill missing dates with NaN values
                full_date_df = pd.DataFrame({'date': date_range})
                combined_data = full_date_df.merge(combined_data, on='date', how='left')
            
            # Get drug columns (exclude date and metadata columns)
            drug_columns = [col for col in combined_data.columns 
                           if col not in ['date', 'source_file', 'Year', 'Month', 'Hour', 'Weekday Name']]
            
            # Check for negative values and invalid data
            for col in drug_columns:
                # Convert to numeric, replacing errors with NaN
                combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
                
                # Check for negative values
                negative_count = (combined_data[col] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"Found {negative_count} negative values in {col}, setting to 0")
                    combined_data[col] = combined_data[col].clip(lower=0)
                
                # Fill missing values with forward fill then backward fill
                combined_data[col] = combined_data[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN, fill with 0
                combined_data[col] = combined_data[col].fillna(0)
            
            self.logger.info(f"Data validation complete. Final dataset: {len(combined_data)} rows, {len(drug_columns)} drugs")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            raise
    
    def prepare_time_series(self, data: pd.DataFrame, drug_name: str) -> pd.DataFrame:
        """Prepare time series data for a specific drug."""
        if drug_name not in data.columns:
            available_drugs = self.get_drug_names(data)
            raise ValueError(f"Drug '{drug_name}' not found in dataset. Available drugs: {available_drugs}")
        
        time_series = data[['date', drug_name]].copy()
        time_series.columns = ['date', 'sales']
        time_series = time_series.dropna().reset_index(drop=True)
        
        # Ensure no negative sales values
        time_series['sales'] = time_series['sales'].clip(lower=0)
        
        return time_series
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and evaluation sets."""
        if len(data) < 30:  # Minimum data requirement
            raise ValueError(f"Insufficient data for splitting. Need at least 30 records, got {len(data)}")
            
        split_index = int(len(data) * train_ratio)
        # Ensure we have at least 10 records for evaluation
        if len(data) - split_index < 10:
            split_index = len(data) - 10
            
        train_data = data.iloc[:split_index].copy()
        eval_data = data.iloc[split_index:].copy()
        
        return train_data, eval_data
    
    def get_drug_names(self, data: pd.DataFrame) -> List[str]:
        """Get list of drug names from the dataset."""
        excluded_columns = ['date', 'source_file', 'Year', 'Month', 'Hour', 'Weekday Name']
        return [col for col in data.columns if col not in excluded_columns]
    
    def get_latest_month(self) -> str:
        """Get the latest month from the sales files."""
        try:
            files = [f for f in os.listdir(self.sales_folder_path) if f.endswith('.csv')]
            # Updated pattern to match sales_YYYY_MM.csv format per instructions
            file_pattern = r'sales_(\d{4})_(\d{2})\.csv'
            
            dates = []
            for file in files:
                match = re.match(file_pattern, file)
                if match:
                    year, month = match.groups()
                    dates.append((int(year), int(month)))
            
            if dates:
                latest_year, latest_month = max(dates)
                return f"{latest_month:02d}_{latest_year}"
            
            return datetime.now().strftime("%m_%Y")
            
        except Exception as e:
            self.logger.error(f"Error getting latest month: {str(e)}")
            return datetime.now().strftime("%m_%Y")
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, any]:
        """Get summary statistics of the dataset."""
        drug_names = self.get_drug_names(data)
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data['date'].min(),
                'end': data['date'].max()
            },
            'total_drugs': len(drug_names),
            'drug_names': drug_names,
            'missing_values': data[drug_names].isnull().sum().to_dict()
        }
        return summary