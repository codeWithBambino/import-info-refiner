# Module Imports
import sys
import os
import argparse
import time 
import pandas as pd
import shutil

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import requirements for pipeline
from src.config.config import CITY_EXTRACTION_PROMPT, INPUT_MANIFESTS, PARTY_STANDARDIZER_PROMPT, PROCESS_MANIFESTS, OUTPUT_CLEANED, REFERENCE_DIR, CLEANED_TEST_DATA_FOLDER, RAW_SHEET_NAME, IMPORT_INFO_TEST_DATASET_URL, PARTY_COLUMNS_TO_STANDARDIZE,ADDRESS_COLUMNS_TO_EXTRACT_CITY, TEMP_DIR
from src.config.folder_name import REMOVE_DUPLICATES_FOLDER, SHIPPER_ENTITY_CLASSIFICATION_FOLDER, CONSIGNEE_CITY_EXTRACTION_FOLDER, LSP_ENTITY_CLASSIFICATION_FOLDER, PIPELINE_MAIN_PROCESS_FOLDER, STANDARDIZE_PARTY_NAMES_FOLDER, CITY_EXTRACTION_FOLDER
from src.helpers.logger import log_message # Imported log_message
from src.helpers.manual_validator import manual_validator
from src.helpers.csv_saver import csv_saver
from src.helpers.google_sheet_handler import read_google_sheet
from src.helpers.standardizer import standardize_data

# Import pipeline functions
# from src.helpers.column_cleaner import column_cleaner
from src.pipeline.duplicate_row_remover import remove_exact_duplicates
from src.pipeline.deduplicator import deduplicate_by_mbl_container
from src.pipeline.scac_mapper import map_scac_to_lsp
from src.pipeline.place_of_receipt_cleaner import standardize_place_of_receipt
from src.pipeline.hs_extractor import extract_hs_code 
from src.pipeline.entity_classifier import classify_entities

# Starting the file processing Life Cycle
def pipeline(test_mode=False):
    start_time = time.time()  # Start timing
    
    # Read test data from Google Sheet
    raw_test_data = read_google_sheet(IMPORT_INFO_TEST_DATASET_URL, RAW_SHEET_NAME)

    # Get list of manifest files
    manifest_files = [
        f for f in os.listdir(INPUT_MANIFESTS)
        if f.endswith(".csv")
    ]

    if not manifest_files:
        print("No manifest files found in the input directory.")
        return

    # Loop through each manifest file
    for raw_manifest_filename in manifest_files:
        input_path = os.path.join(INPUT_MANIFESTS, raw_manifest_filename)
        output_path = CLEANED_TEST_DATA_FOLDER if test_mode else OUTPUT_CLEANED
        processing_filepath = PROCESS_MANIFESTS # Moved here as it's used before the main try block
        temp_path = os.path.join(TEMP_DIR, raw_manifest_filename)
        os.makedirs(PROCESS_MANIFESTS, exist_ok=True) # Moved here

        current_processing_file_path = os.path.join(processing_filepath, raw_manifest_filename)
        final_cleaned_file_path = os.path.join(output_path, raw_manifest_filename)

        try: 
            # Initial setup try-except block
            try:
                if test_mode:
                    dataframe = raw_test_data
                    raw_manifest_filename = "raw_test_dataset.csv" # Ensure this is handled if test_mode changes raw_manifest_filename for paths
                else:
                    dataframe = pd.read_csv(input_path)
                    dataframe['ID'] = list(map(str, range(1, len(dataframe) + 1)))

                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Processing started for {raw_manifest_filename}. Initial Shape: {dataframe.shape}",
                    level="info"
                )
                # dataframe = dataframe.head(500) # Remove later
                
                # Copy initial file to processing directory
                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                
                print(f"\nProcessing {raw_manifest_filename}:")
                print(f"📊 Initial Shape: {dataframe.shape}")
                print(f"💾 Initial file copied to: {processing_filepath}")

                main_dataframe = dataframe.copy()

            except Exception as e:
                error_msg = f"Error during initial setup for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Initial Setup for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file if initial setup fails (will execute outer finally)

            # Step 1: Remove exact duplicates
            try:
                dataframe = remove_exact_duplicates(dataframe, raw_manifest_filename)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 1: Exact Duplicates Removal - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )
                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 1: Exact Duplicates Removal processed | 📊 Shape: {dataframe.shape}")
            except Exception as e:
                error_msg = f"Error during Step 1 (Exact Duplicates Removal) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 1: Exact Duplicates Removal for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)

            # Step 2: Deduplicate by Master BOL + Container No
            try:
                dataframe = deduplicate_by_mbl_container(dataframe, raw_manifest_filename)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 2: MBL+Container Deduplication - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )
                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 2: MBL+Container Deduplication processed | 📊 Shape: {dataframe.shape}")
            except Exception as e:
                error_msg = f"Error during Step 2 (MBL+Container Deduplication) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 2: MBL+Container Deduplication for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)

            # Step 3: Map SCAC to LSP
            try:
                scac_path = os.path.join(REFERENCE_DIR, "scac_codes.csv")
                scac_dataframe = pd.read_csv(scac_path)
                dataframe = map_scac_to_lsp(dataframe, scac_dataframe, raw_manifest_filename)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 3: SCAC to LSP Mapping - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )
                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 3: SCAC to LSP Mapping processed | 📊 Shape: {dataframe.shape}")
            except Exception as e:
                error_msg = f"Error during Step 3 (SCAC to LSP Mapping) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 3: SCAC to LSP Mapping for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)

            # Step 4: Party Standardization
            try:
                dataframe = standardize_data(dataframe = dataframe, raw_manifest_filename=raw_manifest_filename, STANDARDIZER_PROMPT=PARTY_STANDARDIZER_PROMPT, COLUMNS_TO_STANDARDIZE=PARTY_COLUMNS_TO_STANDARDIZE, FOLDER_NAME=STANDARDIZE_PARTY_NAMES_FOLDER, city_flag=False)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 4: Party Standardization - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )

                # Manual validation
                manual_validator("Step 4 Party Names Standardization", main_dataframe, dataframe, column_names=PARTY_COLUMNS_TO_STANDARDIZE)
                print("✅ Manual validation completed for Step 4: Party Standardization")

                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 4: Party Standardization processed | 📊 Shape: {dataframe.shape}")
            
            except Exception as e:
                error_msg = f"Error during Step 4 (Party Standardization) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 4: Party Standardization for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)

            # Step 5: Place of Receipt Standardization
            try:
                dataframe = standardize_place_of_receipt(dataframe, "Place of Receipt", raw_manifest_filename)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 5: Place of Receipt Standardization - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )

                # Manual validation
                manual_validator("Step 5 Place of Receipt Standardization", main_dataframe, dataframe, column_names=["Place of Receipt"])
                print("✅ Manual validation completed for Step 5: Place of Receipt Standardization")

                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 5: Place of Receipt Standardization processed | 📊 Shape: {dataframe.shape}")

            except Exception as e:
                error_msg = f"Error during Step 5 (Place of Receipt Standardization) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 5: Place of Receipt Standardization for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)

            # Step 6: Extract Shipper/Consignee City from address
            try:
                dataframe = standardize_data(dataframe = dataframe, raw_manifest_filename=raw_manifest_filename, STANDARDIZER_PROMPT=CITY_EXTRACTION_PROMPT, COLUMNS_TO_STANDARDIZE=ADDRESS_COLUMNS_TO_EXTRACT_CITY, FOLDER_NAME=CITY_EXTRACTION_FOLDER, city_flag=True)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 6: Shipper City Extraction - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )
                # Manual validation
                manual_validator("Step 6 Shipper / Consignee City Extraction", main_dataframe, dataframe, column_names=["Shipper City", "Consignee City"])
                print("✅ Manual validation completed for Step 6: City Extraction")
                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 6: Shipper City Extraction processed | 📊 Shape: {dataframe.shape}")

            except Exception as e:
                error_msg = f"Error during Step 6 (Shipper City Extraction) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 6: Shipper City Extraction for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)

            # # Step 7: Entity Classification
            # try:
            #     dataframe = classify_entities(dataframe, raw_manifest_filename)
            #     log_message(
            #         folder=PIPELINE_MAIN_PROCESS_FOLDER,
            #         raw_manifest_filename=raw_manifest_filename,
            #         log_string=f"Step 7: Entity Classification - SUCCESS. Shape: {dataframe.shape}",
            #         level="info"
            #     )
            #     # Manual validation
            #     manual_validator("Step 7 Entity Classification", main_dataframe, dataframe, column_names=["Shipper", "Consignee"])
            #     print("✅ Manual validation completed for Step 7: Entity Classification")
            #     csv_saver(dataframe, processing_filepath, raw_manifest_filename)
            #     print(f"✅ Step 7: Entity Classification processed | 📊 Shape: {dataframe.shape}")
            
            # except Exception as e:
            #     error_msg = f"Error during Step 7 (Entity Classification) for '{raw_manifest_filename}': {e}"
            #     print(error_msg)
            #     log_message(
            #         folder=PIPELINE_MAIN_PROCESS_FOLDER,
            #         raw_manifest_filename=raw_manifest_filename,
            #         log_string=f"Step 7: Entity Classification for {raw_manifest_filename} - FAILURE. Error: {e}",
            #         level="error"
            #     )
            #     continue # Skip to the next file (will execute outer finally)

            # Step 11: HS Code Extraction
            try:
                dataframe = extract_hs_code(dataframe, raw_manifest_filename)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 11: HS Code Extraction - SUCCESS. Shape: {dataframe.shape}",
                    level="info"
                )

                csv_saver(dataframe, processing_filepath, raw_manifest_filename)
                print(f"✅ Step 11: HS Code Extraction processed | 📊 Shape: {dataframe.shape}")
            except Exception as e:
                error_msg = f"Error during Step 11 (HS Code Extraction) for '{raw_manifest_filename}': {e}"
                print(error_msg)
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Step 11: HS Code Extraction for {raw_manifest_filename} - FAILURE. Error: {e}",
                    level="error"
                )
                continue # Skip to the next file (will execute outer finally)


            log_message( 
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Processing for {raw_manifest_filename} completed successfully. Ready for moving.",
                level="info"
            )

        finally: 
            if os.path.exists(current_processing_file_path):
                try:
                    os.makedirs(output_path, exist_ok=True) # Ensure destination directory exists
                    shutil.move(current_processing_file_path, final_cleaned_file_path)
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"File '{raw_manifest_filename}' moved from '{current_processing_file_path}' to '{final_cleaned_file_path}'",
                        level="info"
                    )
                    print(f"✅ File '{raw_manifest_filename}' moved to: {final_cleaned_file_path}")
                except Exception as move_e:
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Error moving file '{current_processing_file_path}' to '{final_cleaned_file_path}': {move_e}",
                        level="error"
                    )
                    print(f"❌ Error moving file '{raw_manifest_filename}' from processing to cleaned: {move_e}")
            else:
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"File '{current_processing_file_path}' not found in processing. Cannot move to cleaned.",
                    level="warning"
                )
                print(f"⚠️ File '{raw_manifest_filename}' not found in processing ({current_processing_file_path}). Cannot move.")

            # Remove processing directory
            if os.path.exists(processing_filepath):
                try:
                    shutil.rmtree(processing_filepath)
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Processing directory '{processing_filepath}' removed successfully.",
                        level="info"
                    )
                    print(f"✅ Processing directory '{processing_filepath}' removed successfully.")
                except Exception as rm_e:
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Error removing processing directory '{processing_filepath}': {rm_e}",
                        level="error"
                    )
                    print(f"❌ Error removing processing directory '{processing_filepath}': {rm_e}")

            # Remove temp directory
            if os.path.exists(temp_path):
                try:
                    shutil.rmtree(temp_path)
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Temp directory '{temp_path}' removed successfully.",
                        level="info"
                    )
                    print(f"✅ Temp directory '{temp_path}' removed successfully.")
                except Exception as rm_e:
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Error removing temp directory '{temp_path}': {rm_e}",
                        level="error"
                    )
                    print(f"❌ Error removing temp directory '{temp_path}': {rm_e}")

            else:
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Processing directory '{processing_filepath}' not found. Cannot remove.",
                    level="warning"
                )
                print(f"⚠️ Processing directory '{processing_filepath}' not found. Cannot remove.")
        print(f"✅ Processing for {raw_manifest_filename} completed successfully.")

    end_time = time.time()
    total_time = end_time - start_time

    # Convert to hours, minutes, seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"Total time taken: {hours}h {minutes}m {seconds:.2f}s")
    log_message(
        folder=PIPELINE_MAIN_PROCESS_FOLDER,
        raw_manifest_filename="",
        log_string=f"Pipeline completed successfully. Total time taken: {hours}h {minutes}m {seconds:.2f}s",
        level="info"
    )
    print(f"✅ Pipeline completed successfully.")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process manifest files.")
    # Add an argument for test mode
    parser.add_argument("--test", action="store_true", help="Run in test mode using data from Google Sheets.")
    # Parse the arguments
    args = parser.parse_args()

    # Call the pipeline function with the test_mode argument
    pipeline(test_mode=args.test)