import pandas as pd
from src.helpers.logger import log_message
from src.config.folder_name import REMOVE_DUPLICATES_FOLDER


def remove_exact_duplicates(df: pd.DataFrame, raw_manifest_filename: str) -> pd.DataFrame:
    """
    Removes exact duplicate rows from the provided DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing shipment manifest data.
        raw_manifest_filename (str): Name of the original input CSV file (for logging).

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.

    Raises:
        ValueError: If input DataFrame is empty or None.
        Exception: For other processing errors.
    """
    try:
        # Validate input DataFrame
        if df is None or df.empty:
            error_msg = "Input DataFrame is empty or None"
            log_message(
                folder=REMOVE_DUPLICATES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=error_msg,
                level="error"
            )
            raise ValueError(error_msg)

        try:
            original_row_count = df.shape[0]
            cleaned_df = df.drop_duplicates(keep='first').reset_index(drop=True)
            cleaned_row_count = cleaned_df.shape[0]
            duplicates_removed = original_row_count - cleaned_row_count

            # Logging
            log_message(
                folder=REMOVE_DUPLICATES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=(
                    f"Step 1 - Exact Duplicate Removal:\n"
                    f"Input Rows: {original_row_count}\n"
                    f"Removed Duplicates: {duplicates_removed}\n"
                    f"Remaining Rows: {cleaned_row_count}"
                )
            )

            return cleaned_df

        except Exception as e:
            error_msg = f"Error during duplicate removal: {str(e)}"
            log_message(
                folder=REMOVE_DUPLICATES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=error_msg,
                level="error"
            )
            raise

    except Exception as e:
        error_msg = f"Unexpected error in remove_exact_duplicates: {str(e)}"
        log_message(
            folder=REMOVE_DUPLICATES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=error_msg,
            level="error"
        )
        raise