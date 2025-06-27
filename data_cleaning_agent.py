import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataCleaningAgent:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.folders = ['dev_activity', 'sentiment', 'supply_demand', 'whale_effect']
        self.results = {}
        self.cleaned_dir = self.base_dir / 'cleaned_data'
        self.cleaned_dir.mkdir(exist_ok=True)
        
        # Create cleaned data directories for each folder
        for folder in self.folders:
            (self.cleaned_dir / folder).mkdir(exist_ok=True)
        
        # Store all loaded datasets
        self.all_datasets = {}
        # Store common tokens
        self.common_tokens = []
        # Store token integrity assessment results
        self.token_integrity = {}
    
    def find_csv_files(self):
        all_files = {}
        for folder in self.folders:
            folder_path = self.base_dir / 'original_data' / folder
            if folder_path.exists():
                files = list(folder_path.glob('*.csv'))
                all_files[folder] = files
        return all_files
    
    def load_data(self, file_path):
        """
        Load CSV file data
        """
        try:
            # Assume first column is date, set as index
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_token_names(self, column_names):
        """
        Extract token names from column names
        
        Args:
        column_names: List of column names
        
        Returns:
        token_names: List of extracted token names
        """
        token_names = []
        for col in column_names:
            # Skip date columns
            if col.lower() == 'datetime' or col.lower() == 'date':
                continue
            
            # Extract token name (part before first underscore)
            parts = col.split('_')
            if len(parts) > 1:
                token_name = parts[0]
                token_names.append(token_name)
        
        return list(set(token_names))  # Return unique token names
    
    def identify_common_tokens(self):
        """
        Identify common tokens that exist in all datasets after cleaning
        
        Returns:
        common_tokens: List of tokens that exist in all datasets
        token_counts: Number of occurrences of each token in different datasets
        """
        print("Starting to identify common tokens in cleaned data...")
        
        token_counts = {}
        
        # Count occurrences of each token in different datasets
        for dataset_key, df in self.all_datasets.items():
            if df is not None and not df.empty:
                # Extract token names from column names
                tokens = self.extract_token_names(df.columns)
                
                # Count occurrences of each token
                for token in tokens:
                    if token in token_counts:
                        token_counts[token] += 1
                    else:
                        token_counts[token] = 1
        
        # Get tokens that exist in all datasets
        self.common_tokens = self.get_tokens_in_all_datasets(token_counts)
        
        return self.common_tokens, token_counts
    
    def assess_data_integrity(self):
        """
        Assess data integrity of each token across datasets
        
        Returns:
        token_integrity: Dictionary containing integrity assessment for each token
        """
        print("Starting data integrity assessment...")
        
        # Initialize token integrity assessment results
        self.token_integrity = {}
        
        # Get all unique token names
        all_unique_tokens = set()
        for df in self.all_datasets.values():
            if df is not None and not df.empty:
                # Extract token names from column names
                tokens = self.extract_token_names(df.columns)
                all_unique_tokens.update(tokens)
        
        # Assess integrity of each token
        for token in all_unique_tokens:
            token_data = {
                "datasets_count": 0,  # Number of datasets containing this token
                "total_rows": 0,     # Total rows for this token
                "missing_values": 0,  # Total missing values for this token
                "missing_percentage": 0.0,  # Missing value percentage for this token
                "datasets": []        # Datasets containing this token
            }
            
            # Iterate through all datasets
            for dataset_key, df in self.all_datasets.items():
                if df is not None and not df.empty:
                    # Extract token names from current dataset column names
                    dataset_tokens = self.extract_token_names(df.columns)
                    
                    if token in dataset_tokens:
                        token_data["datasets_count"] += 1
                        token_data["datasets"].append(dataset_key)
                        
                        # Find all columns containing this token
                        token_columns = [col for col in df.columns if col.startswith(token + '_')]
                        
                        # Calculate rows and missing values for this token in current dataset
                        rows = len(df) * len(token_columns)  # Each token may have multiple indicator columns
                        missing = df[token_columns].isna().sum().sum()
                        
                        token_data["total_rows"] += rows
                        token_data["missing_values"] += missing
            
            # Calculate missing value percentage
            if token_data["total_rows"] > 0:
                token_data["missing_percentage"] = (token_data["missing_values"] / token_data["total_rows"]) * 100
            
            # Store token integrity assessment results
            self.token_integrity[token] = token_data
        
        # Sort by missing value percentage
        sorted_integrity = dict(sorted(self.token_integrity.items(), 
                                      key=lambda x: x[1]["missing_percentage"]))
        
        print(f"Completed data integrity assessment for {len(all_unique_tokens)} tokens")
        return sorted_integrity
    
    def clean_data(self, df, file_name):
        """
        Clean data
        
        Args:
        df: DataFrame to be cleaned
        file_name: File name for recording results
        
        Returns:
        cleaned_df: Cleaned DataFrame
        report: Cleaning report
        """
        if df is None:
            return None, {"error": "Failed to load data"}
        
        # Special handling: Convert amount_in_top_holders data from absolute values to changes
        if "amount_in_top_holders" in file_name and "whale_effect" in file_name:
            print(f"Special processing {file_name}: Converting amount_in_top_holders data from absolute values to changes")
            print(f"First row data before conversion:\n{df.iloc[0]}")
            print(f"Second row data before conversion:\n{df.iloc[1]}")
            
            # Calculate daily changes for each column (today's value minus yesterday's value)
            for col in df.columns:
                if col.endswith('_amount_in_top_holders'):
                    # Calculate daily changes and replace original data
                    print(f"Processing column: {col}")
                    original_values = df[col].copy()
                    df[col] = df[col].diff()
                    # Check if successfully converted to changes
                    if len(df) > 1:
                        expected_diff = original_values.iloc[1] - original_values.iloc[0]
                        actual_diff = df[col].iloc[1]
                        print(f"Column {col} second row change check: expected={expected_diff}, actual={actual_diff}")
            
            # First row becomes NaN, remove it
            df = df.dropna(how='all')
            print(f"Converted {file_name} data to changes, first row data:\n{df.iloc[0]}")
            
            # Ensure data has been converted to changes
            if len(df) > 1:
                # Check if first and second row dates are different
                assert df.iloc[0].name != df.iloc[1].name, "Data not successfully converted to changes - date check failed"
                
                # Check if some column values are indeed changes
                sample_cols = [col for col in df.columns if col.endswith('_amount_in_top_holders')][:3]
                for col in sample_cols:
                    print(f"Checking if column {col} values are changes")
                    print(f"First row: {df[col].iloc[0]}")
                    print(f"Second row: {df[col].iloc[1]}")
            
            print(f"Confirmed {file_name} data has been successfully converted to changes")
            
            # Final check before saving converted data
            print(f"Final check before saving - data shape: {df.shape}")
            if len(df) > 0:
                print(f"First row data sample before saving (first 5 columns):\n{df.iloc[0][df.columns[:5]]}")
            else:
                print("Warning: Converted data is empty!")
        
        original_shape = df.shape
        report = {
            "file_name": file_name,
            "original_shape": original_shape,
            "missing_values_before": df.isna().sum().sum(),
            "missing_percentage_before": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        # 1. Check and handle missing values
        # Calculate missing value percentage for each column
        missing_cols = df.columns[df.isna().mean() > 0.3]  # Columns with >30% missing values
        report["columns_removed_missing"] = len(missing_cols)
        report["columns_removed_missing_names"] = list(missing_cols)
        
        # Remove columns with too many missing values
        df = df.drop(columns=missing_cols)
        
        # 2. Stratified processing of tokens based on zero value ratio
        zero_percentage = (df == 0).mean()

        # Low activity tokens (>70% zero values), remove
        low_activity_cols = zero_percentage[zero_percentage > 0.7].index
        report["columns_removed_low_activity"] = len(low_activity_cols)
        report["columns_removed_low_activity_names"] = list(low_activity_cols)
        df = df.drop(columns=low_activity_cols)

        # High activity tokens (<30% zero values), use directly
        # These columns don't need special processing as they will be handled with other columns in subsequent steps
        
        # 2.1 Handle consecutive zero values by converting them to NaN
        # Define threshold for consecutive zeros (how many consecutive days of zeros are considered abnormal)
        consecutive_zeros_threshold = 5
        
        # Iterate through each column
        for col in df.columns:
            # Create boolean mask marking zero value positions
            is_zero = df[col] == 0
            
            # Calculate length of consecutive zeros
            consecutive_zeros = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumsum()
            
            # Set positions with consecutive zeros exceeding threshold to NaN
            mask = consecutive_zeros >= consecutive_zeros_threshold
            df.loc[mask, col] = np.nan
            
        # Update missing value information in report
        report["missing_values_after_consecutive_zeros_conversion"] = df.isna().sum().sum()
        report["missing_percentage_after_consecutive_zeros_conversion"] = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100

        # 3. Detect outliers (statistics only, no replacement)
        # Use IQR method to detect outliers
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Count outliers
        outliers = ((df < lower_bound) | (df > upper_bound)).sum().sum()
        report["outliers_detected"] = outliers
        
        # 4. Don't fill remaining missing values, use data with missing values directly
        df_cleaned = df.copy()
        # No longer use fillna method to fill missing values
        
        # Update report
        report["missing_values_after"] = df_cleaned.isna().sum().sum()
        report["missing_percentage_after"] = (df_cleaned.isna().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1])) * 100
        report["cleaned_shape"] = df_cleaned.shape
        
        return df_cleaned, report
    
    def analyze_data(self, df, file_name):
        """
        Analyze data robustness
        
        Args:
        df: DataFrame to analyze
        file_name: File name for saving charts
        """
        if df is None or df.empty:
            return {"error": "No data to analyze"}
        
        analysis = {}
        
        # 1. Basic statistics
        analysis["basic_stats"] = df.describe().to_dict()
        
        # 2. Calculate coefficient of variation (CV) for each column as robustness indicator
        cv = df.std() / df.mean()
        analysis["coefficient_of_variation"] = cv.to_dict()
        
        # 3. Calculate skewness and kurtosis for each column
        skewness = df.skew()
        kurtosis = df.kurtosis()
        analysis["skewness"] = skewness.to_dict()
        analysis["kurtosis"] = kurtosis.to_dict()
        
        return analysis
    
    def process_all_files(self):
        """
        Process all files
        """
        all_files = self.find_csv_files()
        all_reports = {}
        cleaned_datasets = {}
        
        # First load all datasets
        for folder, files in all_files.items():
            for file_path in files:
                print(f"Loading {file_path}...")
                df = self.load_data(file_path)
                if df is not None:
                    # Use folder and file name as key to store dataset
                    dataset_key = f"{folder}/{file_path.stem}"
                    self.all_datasets[dataset_key] = df
                    
                    # Special handling: directly process whale_effect/amount_in_top_holders.csv file
                    if folder == "whale_effect" and "amount_in_top_holders" in file_path.name:
                        print(f"\nDirectly processing {file_path}...")
                        print(f"Original data shape: {df.shape}")
                        print(f"Original data first two rows example:\n{df.iloc[:2]}")
                        
                        # Create a new DataFrame to save absolute value data
                        absolute_df = df.copy()
                        
                        # Create a new DataFrame to save change data
                        change_df = df.copy()
                        
                        # Calculate daily changes for each column (today's value minus yesterday's value)
                        for col in change_df.columns:
                            if col.endswith('_amount_in_top_holders'):
                                print(f"Processing column: {col}")
                                original_values = change_df[col].copy()
                                change_df[col] = change_df[col].diff()
                                # Check if successfully converted to changes
                                if len(change_df) > 1:
                                    expected_diff = original_values.iloc[1] - original_values.iloc[0]
                                    actual_diff = change_df[col].iloc[1]
                                    print(f"Column {col} second row change check: expected={expected_diff}, actual={actual_diff}")
                        
                        # First row becomes NaN, remove it
                        change_df = change_df.dropna(how='all')
                        print(f"Converted data in {file_path} to changes, data shape: {change_df.shape}")
                        print(f"First row data after conversion:\n{change_df.iloc[0]}")
                        
                        # Update dataset - use change data as main dataset
                        self.all_datasets[dataset_key] = change_df
                        
                        # Save absolute value data to separate file
                        absolute_output_path = self.cleaned_dir / folder / f"absolute_{file_path.name}"
                        absolute_df.to_csv(absolute_output_path)
                        print(f"Saved absolute value data to: {absolute_output_path}")
                    

        
        # Assess original data integrity
        token_integrity = self.assess_data_integrity()
        
        # Process each file
        for folder, files in all_files.items():
            folder_reports = []
            for file_path in files:
                print(f"Processing {file_path}...")
                dataset_key = f"{folder}/{file_path.stem}"
                df = self.all_datasets.get(dataset_key)
                
                # Clean data
                cleaned_df, report = self.clean_data(df, file_path.name)
                
                if cleaned_df is not None:
                    # Save cleaned data
                    output_path = self.cleaned_dir / folder / file_path.name
                    
                    # For amount_in_top_holders.csv file, change data has been processed and saved earlier
                    # Only process other files here
                    if not (folder == "whale_effect" and "amount_in_top_holders" in file_path.name):
                        cleaned_df.to_csv(output_path)
                    
                    # Store cleaned datasets for subsequent common token identification
                    cleaned_datasets[dataset_key] = cleaned_df
                    
                    # Analyze data
                    analysis = self.analyze_data(cleaned_df, file_path)
                    report["analysis"] = analysis
                    
                    # Add token integrity information
                    token_integrity_in_file = {}
                    # Extract token names from column names
                    tokens = self.extract_token_names(cleaned_df.columns)
                    for token in tokens:
                        if token in token_integrity:
                            token_integrity_in_file[token] = token_integrity[token]["missing_percentage"]
                    report["token_integrity"] = token_integrity_in_file
                
                folder_reports.append(report)
            
            all_reports[folder] = folder_reports
        
        # Use cleaned datasets to identify common tokens
        # Temporarily replace original datasets to use existing identify_common_tokens method
        original_datasets = self.all_datasets
        self.all_datasets = cleaned_datasets
        
        # Identify common tokens in cleaned data
        common_tokens, token_counts = self.identify_common_tokens()
        
        # Restore original datasets
        self.all_datasets = original_datasets
        
        # Save cleaning reports
        self.save_reports(all_reports)
        
        # Save token integrity report
        self.save_token_integrity_report(token_integrity)
        
        # Save common token list
        self.save_common_tokens(common_tokens, token_counts)
        
        return all_reports
    
    def save_token_integrity_report(self, token_integrity):
        """
        Save token integrity report
        """
        # Convert token integrity assessment results to DataFrame
        rows = []
        for token, data in token_integrity.items():
            row = {
                "token": token,
                "datasets_count": data["datasets_count"],
                "total_rows": data["total_rows"],
                "missing_values": data["missing_values"],
                "missing_percentage": f"{data['missing_percentage']:.2f}%",
                "datasets": ", ".join(data["datasets"])
            }
            rows.append(row)
        
        integrity_df = pd.DataFrame(rows)
        integrity_df.to_csv(self.cleaned_dir / "token_integrity_report.csv", index=False)
        print(f"Token integrity report saved to {self.cleaned_dir / 'token_integrity_report.csv'}")
    
    def get_tokens_in_all_datasets(self, token_counts):
        """
        Get tokens that exist in all datasets
        
        Args:
        token_counts: Number of occurrences of each token in different datasets
        
        Returns:
        all_datasets_tokens: List of tokens that exist in all datasets
        """
        # Get total number of datasets
        if hasattr(self, 'total_datasets'):
            total_datasets = self.total_datasets
        elif hasattr(self, 'dataset_count_by_folder'):
            total_datasets = sum(self.dataset_count_by_folder.values())
        else:
            total_datasets = len(self.all_datasets)
        
        # Filter tokens that exist in all datasets
        all_datasets_tokens = [token for token, count in token_counts.items() if count == total_datasets]
        print(f"Found {len(all_datasets_tokens)} tokens that exist in all {total_datasets} datasets")
        
        return all_datasets_tokens
    
    def save_common_tokens(self, common_tokens, token_counts):
        """
        Save common token list that exists in all datasets in cleaned data
        """
        # Save common token list
        with open(self.cleaned_dir / "common_tokens.txt", "w") as f:
            f.write("Common Token List in All Datasets After Cleaning\n")
            f.write("==========================================\n\n")
            f.write(f"Number of Common Tokens: {len(common_tokens)}\n\n")
            
            f.write("Token List:\n")
            for token in sorted(common_tokens):
                f.write(f"- {token}\n")
        
        print(f"Common token list for cleaned data saved to {self.cleaned_dir / 'common_tokens.txt'}")
    
    def save_reports(self, reports):
        """
        Save cleaning reports
        """
        # Convert reports to DataFrame and save as CSV
        all_rows = []
        for folder, folder_reports in reports.items():
            for report in folder_reports:
                row = {
                    "folder": folder,
                    "file_name": report["file_name"],
                    "original_shape": f"{report['original_shape'][0]}x{report['original_shape'][1]}",
                    "cleaned_shape": f"{report.get('cleaned_shape', (0, 0))[0]}x{report.get('cleaned_shape', (0, 0))[1]}",
                    "columns_removed_missing": report.get("columns_removed_missing", 0),
                    "columns_removed_low_activity": report.get("columns_removed_low_activity", 0),
                    "missing_values_before": report.get("missing_values_before", 0),
                    "missing_percentage_before": f"{report.get('missing_percentage_before', 0):.2f}%",
                    "missing_values_after_consecutive_zeros": report.get("missing_values_after_consecutive_zeros_conversion", 0),
                    "missing_percentage_after_consecutive_zeros": f"{report.get('missing_percentage_after_consecutive_zeros_conversion', 0):.2f}%",
                    "missing_values_after": report.get("missing_values_after", 0),
                    "missing_percentage_after": f"{report.get('missing_percentage_after', 0):.2f}%",
                    "outliers_detected": report.get("outliers_detected", 0)
                }
                all_rows.append(row)
        
        report_df = pd.DataFrame(all_rows)
        report_df.to_csv(self.cleaned_dir / "cleaning_report.csv", index=False)
        
        # Generate summary report
        with open(self.cleaned_dir / "summary_report.txt", "w") as f:
            f.write("Data Cleaning Summary Report\n")
            f.write("=================\n\n")
            
            # Add common token information
            f.write(f"Number of Common Tokens: {len(self.common_tokens)}\n")
            if len(self.common_tokens) > 0:
                f.write("Common Token Examples: " + ", ".join(self.common_tokens[:10]))
                if len(self.common_tokens) > 10:
                    f.write(" etc...")
            f.write("\n\n")
            
            for folder, folder_reports in reports.items():
                f.write(f"Folder: {folder}\n")
                f.write("-" * 50 + "\n")
                
                for report in folder_reports:
                    f.write(f"File: {report['file_name']}\n")
                    f.write(f"Original size: {report['original_shape'][0]}x{report['original_shape'][1]}\n")
                    f.write(f"Cleaned size: {report.get('cleaned_shape', (0, 0))[0]}x{report.get('cleaned_shape', (0, 0))[1]}\n")
                    f.write(f"Columns removed: {report.get('columns_removed', 0)}\n")
                    f.write(f"Missing values before cleaning: {report.get('missing_values_before', 0)} ({report.get('missing_percentage_before', 0):.2f}%)\n")
                    if 'missing_values_after_consecutive_zeros_conversion' in report:
                        f.write(f"Missing values after consecutive zeros to NaN conversion: {report.get('missing_values_after_consecutive_zeros_conversion', 0)} ({report.get('missing_percentage_after_consecutive_zeros_conversion', 0):.2f}%)\n")
                    f.write(f"Missing values after cleaning: {report.get('missing_values_after', 0)} ({report.get('missing_percentage_after', 0):.2f}%)\n")
                    f.write(f"Outliers detected: {report.get('outliers_detected', 0)}\n\n")
                
                f.write("\n")

# Usage example
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agent = DataCleaningAgent(base_dir)
    reports = agent.process_all_files()
    print("Data cleaning completed! Reports have been saved to the cleaned_data directory.")