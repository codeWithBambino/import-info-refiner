import pandas as pd
from src.config.folder_name import REMOVE_DUPLICATES_FOLDER # Corrected import
from src.helpers.logger import log_message
import os

def csv_saver(df: pd.DataFrame, output_path: str, raw_manifest_filename: str):
    """
    Saves DataFrame to CSV.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Destination directory for the CSV.
        raw_manifest_filename (str): Original CSV file name, used as the save name.
    """
    try:
        # Removed prefix logic
        # filename_with_prefix = f"{prefix}{raw_manifest_filename}"
        full_file_path = os.path.join(output_path, raw_manifest_filename) # Save with original name

        df.to_csv(full_file_path, index=False)
        log_message(
            folder=REMOVE_DUPLICATES_FOLDER, 
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"DataFrame successfully saved to: {full_file_path}"
        )
    except Exception as e:
        log_message(
            folder=REMOVE_DUPLICATES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"ERROR saving DataFrame: {e}",
            level='error'
        )
        raise