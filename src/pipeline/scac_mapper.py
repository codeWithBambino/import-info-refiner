import pandas as pd
from src.helpers.logger import log_message
from src.config.folder_name import MAP_SCAC_FOLDER


def map_scac_to_lsp(df: pd.DataFrame, scac_df: pd.DataFrame, raw_manifest_filename: str) -> pd.DataFrame:
    """
    Maps SCAC codes in 'Carrier Code' column to full LSP names using the SCAC reference file.

    Args:
        df (pd.DataFrame): Manifest DataFrame to process.
        scac_df (pd.DataFrame): SCAC reference table with columns ['SCAC', 'Company name'].
        raw_manifest_filename (str): File name for logging.

    Returns:
        pd.DataFrame: Updated DataFrame with new 'LSP' column mapped from SCAC.

    Raises:
        ValueError: If required columns are missing.
    """
    try:
        required_df_col = "Carrier Code"
        required_ref_cols = {"SCAC", "Company name"}

        if required_df_col not in df.columns:
            raise ValueError(f"Missing '{required_df_col}' column in manifest data.")

        if not required_ref_cols.issubset(scac_df.columns):
            raise ValueError(f"SCAC mapping must include columns: {', '.join(required_ref_cols)}")

        # Normalize for matching
        df["Carrier Code"] = df["Carrier Code"].astype(str).str.strip().str.upper()
        scac_df["SCAC"] = scac_df["SCAC"].astype(str).str.strip().str.upper()
        scac_df["Company name"] = scac_df["Company name"].astype(str).str.strip()

        # Log SCAC match debug
        unique_df = set(df["Carrier Code"].dropna().unique())
        unique_ref = set(scac_df["SCAC"].dropna().unique())
        matched = unique_df.intersection(unique_ref)

        log_message(
            folder=MAP_SCAC_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"SCAC Debug Info:\n"
                f"  Manifest SCACs: {len(unique_df)}\n"
                f"  Reference SCACs: {len(unique_ref)}\n"
                f"  Matched SCACs: {len(matched)}"
            )
        )

        # Merge and map
        merged_df = df.merge(scac_df, how="left", left_on="Carrier Code", right_on="SCAC")
        merged_df["LSP"] = merged_df["Company name"]
        merged_df.drop(columns=["Country", "Company name", "SCAC"], inplace=True)

        unmapped_count = merged_df["LSP"].isna().sum()
        log_message(
            folder=MAP_SCAC_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Step 3 - SCAC Mapping completed. Unmapped rows: {unmapped_count}"
        )

        return merged_df

    except Exception as e:
        log_message(
            folder=MAP_SCAC_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Error during SCAC mapping: {e}",
            level="error"
        )
        raise