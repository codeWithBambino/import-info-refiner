import pandas as pd
import os
from src.config.config import MANUAL_VALIDATION_DIR

try:
    from src.config.config import MANUAL_VALIDATION_DIR
except ImportError:
    print("Warning: Could not import MANUAL_VALIDATION_DIR from src.config.config.")
    print("Using a default directory 'manual_validation_output' in the current directory.")



def manual_validator(step_name: str,
                     raw_dataframe: pd.DataFrame,
                     cleaned_dataframe: pd.DataFrame,
                     column_names: list[str] = None,
                     left_on: list[str] = None,
                     right_on: list[str] = None):
    """
    Compares two pandas DataFrames based on a common 'ID' index and
    identifies differences in specified or all columns, saving the results
    for manual validation.

    Args:
        step_name (str): Name of the processing step. Used for the output filename.
        raw_dataframe (pd.DataFrame): The first DataFrame (e.g., raw data).
                                     Must contain a 'ID' column.
        cleaned_dataframe (pd.DataFrame): The second DataFrame (e.g., cleaned data).
                                        Must contain a 'ID' column.
        column_names (list[str], optional): List of columns to compare. If None or empty,
                                            all common columns (except 'ID') are compared.
        left_on (list[str], optional): List of column names from raw_dataframe to compare.
                                       Must be used together with right_on and be of same length.
        right_on (list[str], optional): List of column names from cleaned_dataframe to compare.
                                        Must be used together with left_on and be of same length.
    Raises:
        ValueError: If 'ID' column is missing in either DataFrame or if left_on/right_on parameters are invalid.
        Exception: Catches and prints any other errors during processing or saving.
    """
    try:
        # Create manual validation directory if it doesn't exist
        manual_validation_dir = MANUAL_VALIDATION_DIR
        os.makedirs(manual_validation_dir, exist_ok=True)

        # Format step name for filename
        formatted_step_name = step_name.lower().replace(' ', '_')
        output_path = os.path.join(manual_validation_dir, f"{formatted_step_name}.csv")

        # Validate input parameters
        if left_on is not None and right_on is None:
            raise ValueError("If left_on is provided, right_on must also be provided.")
        if right_on is not None and left_on is None:
            raise ValueError("If right_on is provided, left_on must also be provided.")
        if left_on is not None and right_on is not None and len(left_on) != len(right_on):
            raise ValueError("left_on and right_on must have the same length.")
        
        # Validate column existence if left_on/right_on are provided
        if left_on is not None:
            missing_left = [col for col in left_on if col not in raw_dataframe.columns]
            if missing_left:
                raise ValueError(f"Columns {missing_left} not found in raw_dataframe.")
        if right_on is not None:
            missing_right = [col for col in right_on if col not in cleaned_dataframe.columns]
            if missing_right:
                raise ValueError(f"Columns {missing_right} not found in cleaned_dataframe.")

        if 'ID' not in raw_dataframe.columns or 'ID' not in cleaned_dataframe.columns:
            raise ValueError("Both DataFrames must contain a 'ID' column.")

        # Ensure 'ID' is set as the index for both DataFrames
        # Using .copy() to avoid modifying the original DataFrames
        df1_indexed = raw_dataframe.set_index('ID').copy()
        df2_indexed = cleaned_dataframe.set_index('ID').copy()

        # Align the DataFrames based on their index ('ID').
        # 'inner' join ensures we only compare rows present in both DataFrames.
        # axis=0 specifies alignment along the index.
        df1_aligned, df2_aligned = df1_indexed.align(df2_indexed, join='inner', axis=0)

        # Initialize a list to collect differences
        differences = []

        # Handle different comparison scenarios
        if left_on is not None and right_on is not None:
            # Compare specific columns with different names
            for left_col, right_col in zip(left_on, right_on):
                if left_col in df1_aligned.columns and right_col in df2_aligned.columns:
                    # Compare values between corresponding columns
                    diff_mask = df1_aligned[left_col] != df2_aligned[right_col]
                    
                    # Get the differing row indices (which are the original 'ID' values)
                    differing_row_ids = df1_aligned[diff_mask].index
                    
                    # Append differences to the list with both column names
                    column_pair = f"{left_col} / {right_col}"
                    for row_id in differing_row_ids:
                        raw_value = df1_aligned.at[row_id, left_col]
                        cleaned_value = df2_aligned.at[row_id, right_col]
                        differences.append((row_id, column_pair, raw_value, cleaned_value))
        else:
            # Determine columns to compare (old logic for same column names)
            if column_names:
                # Filter requested columns to only include those present in both aligned dataframes
                cols_to_compare = [col for col in column_names if col in df1_aligned.columns and col in df2_aligned.columns]
                if not cols_to_compare:
                    print(f"Warning: None of the specified columns {column_names} found in both dataframes after alignment. No comparison performed.")
                    # Create an empty DataFrame with the expected output columns
                    result_df = pd.DataFrame(columns=['Index', 'Column', 'Raw', 'Cleaned', 'Count'])
                    # Save the empty DataFrame and return
                    result_df.to_csv(output_path, index=False)
                    return # Exit the function early
            else:
                # Compare all columns present in both aligned dataframes (excluding the index)
                cols_to_compare = df1_aligned.columns # df1_aligned and df2_aligned have same columns after align(join='inner')

            # Iterate over each column to compare (standard comparison for same column names)
            for column in cols_to_compare:
                # Compare the columns from the aligned DataFrames
                diff_mask = df1_aligned[column] != df2_aligned[column]
                
                # Get the differing row indices (which are the original 'ID' values)
                differing_row_ids = df1_aligned[diff_mask].index
                
                # Append differences to the list
                for row_id in differing_row_ids:
                    raw_value = df1_aligned.at[row_id, column]
                    cleaned_value = df2_aligned.at[row_id, column]
                    differences.append((row_id, column, raw_value, cleaned_value))

        # Create a DataFrame from the differences
        diff_df = pd.DataFrame(differences, columns=['Index', 'Column', 'Raw', 'Cleaned'])

        # If there are differences, calculate the count and format the output
        if not diff_df.empty:
            # Calculate the count of each (Raw, Cleaned) pair
            # This count represents how many times a specific raw value was mapped to a specific cleaned value for a given column
            count_series = diff_df.groupby(['Column', 'Raw', 'Cleaned']).size().reset_index(name='Count')

            # Merge the count back into the diff_df
            result_df = pd.merge(diff_df, count_series, on=['Column', 'Raw', 'Cleaned'])

            # Drop duplicates based on the combination of Column, Raw, Cleaned, and Count
            # This keeps one row for each unique type of difference (e.g., 'abc' -> 'def' occurred 5 times in column 'X')
            result_df.drop_duplicates(subset=['Column', 'Raw', 'Cleaned', 'Count'], inplace=True)

        else:
            # If no differences, create an empty DataFrame with the expected output columns
            result_df = pd.DataFrame(columns=['Index', 'Column', 'Raw', 'Cleaned', 'Count'])

        # Save the resulting DataFrame to CSV
        result_df.to_csv(output_path, index=False)
        print(f"Differences saved to {output_path}")

    except ValueError as ve:
        print(f"ValueError in manual_validator for {step_name}: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred in manual_validator for {step_name}: {e}")