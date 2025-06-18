import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from pathlib import Path
import time

def compile_eye_data(input_dir="data", output_file=None):
    """
    Combines all eye_image_data_*.csv files into a single large CSV file
    
    Args:
        input_dir: Directory where the CSV files are located
        output_file: Name of the output file (default creates a timestamped file)
    
    Returns:
        Path to the created file
    """
    # Create default output filename if not provided
    if output_file is None:
        output_file = f"eye_image_data.csv"
    
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return None
    
    # Find all eye data CSV files
    pattern = os.path.join(input_dir, "eye_image_data_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No eye data CSV files found in {input_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to combine")
    
    # Read and combine all CSV files
    all_dataframes = []
    total_blinks = 0
    total_frames = 0
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            frames = len(df)
            blinks = df['manual_blink'].sum()
            
            print(f"Reading {os.path.basename(file_path)}: {frames} frames, {blinks} blinks")
            
            all_dataframes.append(df)
            total_frames += frames
            total_blinks += blinks
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_dataframes:
        print("No valid data found in CSV files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to the output file
    output_path = os.path.join(input_dir, output_file)
    combined_df.to_csv(output_path, index=False)
    
    # Calculate statistics
    blink_percentage = (total_blinks / total_frames) * 100 if total_frames > 0 else 0
    
    print("\n===== Dataset Summary =====")
    print(f"Total frames: {total_frames}")
    print(f"Total blinks: {total_blinks}")
    print(f"Blink percentage: {blink_percentage:.2f}%")
    print(f"Combined dataset saved to: {output_path}")
    print(f"Dataset shape: {combined_df.shape}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine multiple eye image CSV files into one dataset")
    parser.add_argument("--input", "-i", default="data", help="Input directory containing CSV files")
    parser.add_argument("--output", "-o", help="Output CSV file name (default: timestamped file)")
    
    args = parser.parse_args()
    
    compile_eye_data(args.input, args.output)