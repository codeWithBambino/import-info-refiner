import pandas as pd
from src.helpers.logger import log_message
from src.config.folder_name import DEDUPLICATE_MBL_CONTAINER_FOLDER


def deduplicate_by_mbl_container(df: pd.DataFrame, raw_manifest_filename: str) -> pd.DataFrame:
    """
    Deduplicates manifest rows grouped by Master BOL + Container Numbers using House BOL logic.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from Step 1.
        raw_manifest_filename (str): For logging purposes.

    Returns:
        pd.DataFrame: De-duplicated DataFrame with LCL flag added.

    Raises:
        ValueError: If required columns are missing.
        Exception: For other processing errors.
    """
    try:
        # Validate required columns
        required_cols = ["Master BOL", "Container Numbers", "House BOL"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {', '.join(missing_cols)}"
            log_message(
                folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=error_msg,
                level="error"
            )
            raise ValueError(error_msg)

        df = df.copy()
        df["LCL"] = "No"  # default value

        pre_dedup_count = df.shape[0]
        result_rows = []

        try:
            grouped = df.groupby(["Master BOL", "Container Numbers"])
        except Exception as e:
            error_msg = f"Error during grouping operation: {str(e)}"
            log_message(
                folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=error_msg,
                level="error"
            )
            raise

        # Log summary
        log_message(
            folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Starting the process for {raw_manifest_filename}. Total rows: {pre_dedup_count}"
        )

        for (mbl, container), group in grouped:
            try:
                group_with_hbl = group[group["House BOL"].notna() & (group["House BOL"].astype(str).str.strip() != "")]

                if len(group) == 1:
                    result_rows.append(group.iloc[0])
                elif len(group) == 2:
                    if len(group_with_hbl) == 1:
                        # Only one row has HBL → keep that
                        row = group_with_hbl.iloc[0].copy()
                        row.loc["LCL"] = "Yes"
                        result_rows.append(row)
                    else:
                        # Keep both
                        for idx, row in group.iterrows():
                            row_copy = row.copy()
                            row_copy.loc["LCL"] = "Yes"
                            result_rows.append(row_copy)
                else:
                    # Keep only those with HBL
                    for idx, row in group_with_hbl.iterrows():
                        row_copy = row.copy()
                        row_copy.loc["LCL"] = "Yes"
                        result_rows.append(row_copy)
                
                log_message(
                    folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Processed MBL={mbl}, Container={container}"
                )
            except Exception as e:
                error_msg = f"Error processing group MBL={mbl}, Container={container}: {str(e)}"
                log_message(
                    folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=error_msg,
                    level="error"
                )
                raise

        deduped_df = pd.DataFrame(result_rows).reset_index(drop=True)

        log_message(
            folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Deduplication complete. Original shape: {df.shape}, Final shape: {deduped_df.shape}"
        )

        return deduped_df

    except Exception as e:
        error_msg = f"Unexpected error during deduplication: {str(e)}"
        log_message(
            folder=DEDUPLICATE_MBL_CONTAINER_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=error_msg,
            level="error"
        )
        raise