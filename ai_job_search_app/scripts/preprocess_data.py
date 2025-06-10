import pandas as pd
import re
from typing import List

def load_and_clean_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Loads data from a list of CSV files, cleans the text data, standardizes columns,
    and concatenates them into a single DataFrame.

    Args:
        file_paths: A list of string paths to the CSV files.

    Returns:
        A single pandas DataFrame containing the combined and cleaned data.
    """
    all_dfs = []
    for file_path in file_paths:
        try:
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path)
            
            # Standardize the resume text column to 'Resume'
            if 'Resume_str' in df.columns:
                df.rename(columns={'Resume_str': 'Resume'}, inplace=True)
            
            # Check for required columns
            if 'Resume' not in df.columns or 'Category' not in df.columns:
                print(f"Warning: Skipping {file_path} because it lacks 'Resume' or 'Category' columns.")
                continue

            # Clean the text data
            df.dropna(subset=['Resume', 'Category'], inplace=True)
            df['cleaned_resume'] = df['Resume'].apply(lambda x: re.sub(r'[\\r\\n\\t]', ' ', str(x)).lower())
            
            # Keep only the necessary columns
            all_dfs.append(df[['cleaned_resume', 'Category']])
            print(f"Loaded and cleaned {len(df)} records from {file_path}.")

        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    if not all_dfs:
        raise ValueError("Could not load any valid data from the provided paths.")

    # Concatenate all dataframes into one
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\\nTotal combined and cleaned records for training: {len(combined_df)}")
    return combined_df

if __name__ == "__main__":
    # Example of how to run the preprocessing with multiple files
    data_paths = [
        "../../Resume.csv",
        "../../UpdatedResumeDataSet.csv"
    ]
    
    combined_data = load_and_clean_data(data_paths)
    print("\\n--- Combined Data Head ---")
    print(combined_data.head())
    print(f"\\nTotal rows: {len(combined_data)}")
    print(f"Categories found: {combined_data['Category'].nunique()}")

    # In the next steps, we will save the processed data
 