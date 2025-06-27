#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data filter script: Extract the data of 36 common tokens from all CSV files in the cleaned_data folder
"""

import os
import pandas as pd
from pathlib import Path
import shutil

def load_common_tokens(file_path):
    """load common tokens list"""
    common_tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # skip the introduction, and get the token list
    start_reading = False
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            start_reading = True
            token = line[2:].strip()  # remove the '-'
            common_tokens.append(token)
        elif start_reading and not line.startswith('- ') and line:
            break
    
    return common_tokens

def get_matching_columns(df_columns, common_tokens):
    """找到与共同token匹配的列名"""
    matching_columns = ['datetime']  # keep datetime column always
    
    for token in common_tokens:
        # search for the complete matching column
        for col in df_columns:
            if col != 'datetime' and token in col:
                # make sure it is a complete match (avoid partial match)
                # for example: avoid matching '0x_dev_activity_1d' to '0x'
                if col.startswith(f'{token}_') or col.endswith(f'_{token}') or col == token:
                    matching_columns.append(col)
                    break  # every token only match one column
    
    return matching_columns

def filter_csv_file(input_path, output_path, common_tokens):
    """filter the csv file, only keep the data of common tokens"""
    try:
        df = pd.read_csv(input_path)
        
        # get the matching columns
        matching_columns = get_matching_columns(df.columns, common_tokens)
        
        # filter the data
        filtered_df = df[matching_columns]
        
        # make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # save the filtered data
        filtered_df.to_csv(output_path, index=False)
        
        print(f"processed: {input_path} -> {output_path}")
        print(f"  columns before: {len(df.columns)}, columns after: {len(filtered_df.columns)}")
        print(f"  match token columns: {[col for col in matching_columns if col != 'datetime']}")
        print()
        
        return True
        
    except Exception as e:
        print(f"error when processing {input_path} : {str(e)}")
        return False

def filter_price_data(input_path, output_path, common_tokens):
    """filter price data file, only keep the price data of common tokens"""
    try:
        # load price data
        df = pd.read_csv(input_path)
        
        # keep datetime column and common tokens columns
        columns_to_keep = ['datetime']
        
        # check each common token if it is in the price data columns
        for token in common_tokens:
            if token in df.columns:
                columns_to_keep.append(token)
        
        # filter the data
        filtered_df = df[columns_to_keep]
        
        # make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # keep filtered data to csv file
        filtered_df.to_csv(output_path, index=False)
        
        print(f"processed price data: {input_path} -> {output_path}")
        print(f"  initial tokens: {len(df.columns) - 1}, tokens after filtered: {len(filtered_df.columns) - 1}")
        print(f"  kept token: {[col for col in columns_to_keep if col != 'datetime']}")
        print()
        
        return True
        
    except Exception as e:
        print(f"error when processing {input_path} : {str(e)}")
        return False

def main():
    # set paths
    base_dir = Path('/Users/queenwen/Desktop/QI_Paper')
    cleaned_data_dir = base_dir / 'cleaned_data'
    price_data_dir = base_dir / 'price_data'
    common_tokens_file = cleaned_data_dir / 'common_tokens.txt'
    output_dir = base_dir / 'filtered_common_tokens_data'
    
    # load common tokens list
    print("load common tokens list...")
    common_tokens = load_common_tokens(common_tokens_file)
    print(f"common tokens: {len(common_tokens)}")
    print(f"Token list: {common_tokens}")
    print()
    
    # create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # process all csv files in subfolders
    processed_files = 0
    failed_files = 0
    
    for subfolder in cleaned_data_dir.iterdir():
        if subfolder.is_dir():
            print(f"process subfolder: {subfolder.name}")
            
            # create the corresponding output subfolder
            output_subfolder = output_dir / subfolder.name
            output_subfolder.mkdir(exist_ok=True)
            
            # process all csv files in the subfolder
            for csv_file in subfolder.glob('*.csv'):
                input_path = csv_file
                output_path = output_subfolder / csv_file.name
                
                if filter_csv_file(input_path, output_path, common_tokens):
                    processed_files += 1
                else:
                    failed_files += 1
    
    # process price data
    if price_data_dir.exists():
        print("processing price data...")
        
        # create the price data output folder
        price_output_dir = output_dir / 'price_data'
        price_output_dir.mkdir(exist_ok=True)
        
        # create token_prices.csv file
        price_file = price_data_dir / 'token_prices.csv'
        if price_file.exists():
            output_price_file = price_output_dir / 'token_prices.csv'
            if filter_price_data(price_file, output_price_file, common_tokens):
                processed_files += 1
            else:
                failed_files += 1
    
    # copy non-csv files to output directory
    for item in cleaned_data_dir.iterdir():
        if item.is_file() and not item.name.endswith('.csv'):
            shutil.copy2(item, output_dir / item.name)
            print(f"copied: {item.name}")
    
    print(f"\n process completed!")
    print(f"processed files: {processed_files}")
    print(f"failed files: {failed_files}")
    print(f"output direction: {output_dir}")

if __name__ == '__main__':
    main()