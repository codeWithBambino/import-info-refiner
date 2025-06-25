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
from src.helpers.city_verifier import city_verifier

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
    
    # Combined output filename
    combined_output_filename = "combined_processed_data.csv"
    
    try:
        # Initialize combined dataframe and file tracking
        combined_dataframe = pd.DataFrame()
        processed_files = []
        
        if test_mode:
            # Read test data from Google Sheet
            raw_test_data = read_google_sheet(IMPORT_INFO_TEST_DATASET_URL, RAW_SHEET_NAME)
            raw_test_data['source_file'] = "raw_test_dataset.csv"
            raw_test_data['ID'] = list(map(str, range(1, len(raw_test_data) + 1)))
            combined_dataframe = raw_test_data
            processed_files = ["raw_test_dataset.csv"]
            
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Test mode: Loaded data from Google Sheets. Shape: {combined_dataframe.shape}",
                level="info"
            )
        else:
            # Get list of manifest files
            manifest_files = [
                f for f in os.listdir(INPUT_MANIFESTS)
                if f.endswith(".csv")
            ]

            if not manifest_files:
                print("No manifest files found in the input directory.")
                return

            print(f"Found {len(manifest_files)} CSV files to process:")
            for file in manifest_files:
                print(f"  - {file}")

            # Read and combine all CSV files
            dataframes_to_combine = []
            for raw_manifest_filename in manifest_files:
                try:
                    input_path = os.path.join(INPUT_MANIFESTS, raw_manifest_filename)
                    df = pd.read_csv(input_path)
                    df['source_file'] = raw_manifest_filename  # Track source file
                    df['ID'] = list(map(str, range(1, len(df) + 1)))
                    dataframes_to_combine.append(df)
                    processed_files.append(raw_manifest_filename)
                    print(f"✅ Loaded {raw_manifest_filename} - Shape: {df.shape}")
                    
                except Exception as e:
                    error_msg = f"Error reading file '{raw_manifest_filename}': {e}"
                    print(f"❌ {error_msg}")
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=error_msg,
                        level="error"
                    )
                    continue

            if not dataframes_to_combine:
                print("No valid CSV files could be loaded.")
                return

            # Combine all dataframes
            combined_dataframe = pd.concat(dataframes_to_combine, ignore_index=True)
            
            # Reassign global IDs after combining
            combined_dataframe['ID'] = list(map(str, range(1, len(combined_dataframe) + 1)))

        print(f"\n📊 Combined Dataset Shape: {combined_dataframe.shape}")
        print(f"📁 Files processed: {len(processed_files)}")
        
        log_message(
            folder=PIPELINE_MAIN_PROCESS_FOLDER,
            raw_manifest_filename=combined_output_filename,
            log_string=f"Combined dataset created. Shape: {combined_dataframe.shape}, Files: {len(processed_files)}",
            level="info"
        )

        # Copy original files to processing folder (only in non-test mode)
        if not test_mode:
            os.makedirs(PROCESS_MANIFESTS, exist_ok=True)
            for filename in processed_files:
                try:
                    source_path = os.path.join(INPUT_MANIFESTS, filename)
                    dest_path = os.path.join(PROCESS_MANIFESTS, filename)
                    shutil.copy2(source_path, dest_path)
                    print(f"📋 Copied {filename} to processing folder")
                except Exception as e:
                    print(f"❌ Error copying {filename}: {e}")
                    log_message(
                        folder=PIPELINE_MAIN_PROCESS_FOLDER,
                        raw_manifest_filename=filename,
                        log_string=f"Error copying file to processing: {e}",
                        level="error"
                    )

        # Store original dataframe for manual validation
        main_dataframe = combined_dataframe.copy()
        dataframe = combined_dataframe.copy()

        # Processing pipeline starts here
        print(f"\n🚀 Starting processing pipeline...")

        # Step 1: Remove exact duplicates
        try:
            dataframe = remove_exact_duplicates(dataframe, combined_output_filename)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 1: Exact Duplicates Removal - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )
            print(f"✅ Step 1: Exact Duplicates Removal processed | 📊 Shape: {dataframe.shape}")
        except Exception as e:
            error_msg = f"Error during Step 1 (Exact Duplicates Removal): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 1: Exact Duplicates Removal - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # Step 2: Deduplicate by Master BOL + Container No
        try:
            dataframe = deduplicate_by_mbl_container(dataframe, combined_output_filename)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 2: MBL+Container Deduplication - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )
            print(f"✅ Step 2: MBL+Container Deduplication processed | 📊 Shape: {dataframe.shape}")
        except Exception as e:
            error_msg = f"Error during Step 2 (MBL+Container Deduplication): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 2: MBL+Container Deduplication - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # Step 3: Map SCAC to LSP
        try:
            scac_path = os.path.join(REFERENCE_DIR, "scac_codes.csv")
            scac_dataframe = pd.read_csv(scac_path)
            dataframe = map_scac_to_lsp(dataframe, scac_dataframe, combined_output_filename)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 3: SCAC to LSP Mapping - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )
            print(f"✅ Step 3: SCAC to LSP Mapping processed | 📊 Shape: {dataframe.shape}")
        except Exception as e:
            error_msg = f"Error during Step 3 (SCAC to LSP Mapping): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 3: SCAC to LSP Mapping - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # Step 4: Party Standardization
        try:
            dataframe = standardize_data(dataframe=dataframe, raw_manifest_filename=combined_output_filename, STANDARDIZER_PROMPT=PARTY_STANDARDIZER_PROMPT, COLUMNS_TO_STANDARDIZE=PARTY_COLUMNS_TO_STANDARDIZE, FOLDER_NAME=STANDARDIZE_PARTY_NAMES_FOLDER, city_flag=False)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 4: Party Standardization - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )

            # Manual validation
            manual_validator("Step 4 Party Names Standardization", main_dataframe, dataframe, column_names=PARTY_COLUMNS_TO_STANDARDIZE)
            print("✅ Manual validation completed for Step 4: Party Standardization")
            print(f"✅ Step 4: Party Standardization processed | 📊 Shape: {dataframe.shape}")
        
        except Exception as e:
            error_msg = f"Error during Step 4 (Party Standardization): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 4: Party Standardization - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # Step 5: Place of Receipt Standardization
        try:
            dataframe = standardize_place_of_receipt(dataframe, "Place of Receipt", combined_output_filename)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 5: Place of Receipt Standardization - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )

            # Manual validation
            manual_validator("Step 5 Place of Receipt Standardization", main_dataframe, dataframe, column_names=["Place of Receipt"])
            print("✅ Manual validation completed for Step 5: Place of Receipt Standardization")
            print(f"✅ Step 5: Place of Receipt Standardization processed | 📊 Shape: {dataframe.shape}")

        except Exception as e:
            error_msg = f"Error during Step 5 (Place of Receipt Standardization): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 5: Place of Receipt Standardization - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # Step 6: Extract Shipper/Consignee City from address
        try:
            dataframe = standardize_data(dataframe=dataframe, raw_manifest_filename=combined_output_filename, STANDARDIZER_PROMPT=CITY_EXTRACTION_PROMPT, COLUMNS_TO_STANDARDIZE=ADDRESS_COLUMNS_TO_EXTRACT_CITY, FOLDER_NAME=CITY_EXTRACTION_FOLDER, city_flag=True)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 6: City Extraction - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )
            # Manual validation
            manual_validator("Step 6 Shipper / Consignee City Extraction", main_dataframe, dataframe, column_names=["Shipper City", "Consignee City"])
            print("✅ Manual validation completed for Step 6: City Extraction")
            dataframe = city_verifier(dataframe, ["Shipper City", "Consignee City"])
            print("✅ City Verification completed for Step 6: City Extraction")
            print(f"✅ Step 6: City Extraction processed | 📊 Shape: {dataframe.shape}")

        except Exception as e:
            error_msg = f"Error during Step 6 (City Extraction): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 6: City Extraction - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # # Step 7: Entity Classification
        # try:
        #     dataframe = classify_entities(dataframe, combined_output_filename)
        #     log_message(
        #         folder=PIPELINE_MAIN_PROCESS_FOLDER,
        #         raw_manifest_filename=combined_output_filename,
        #         log_string=f"Step 7: Entity Classification - SUCCESS. Shape: {dataframe.shape}",
        #         level="info"
        #     )
        #     # Manual validation
        #     manual_validator("Step 7 Entity Classification", main_dataframe, dataframe, column_names=["Shipper", "Consignee"])
        #     print("✅ Manual validation completed for Step 7: Entity Classification")
        #     print(f"✅ Step 7: Entity Classification processed | 📊 Shape: {dataframe.shape}")
        
        # except Exception as e:
        #     error_msg = f"Error during Step 7 (Entity Classification): {e}"
        #     print(f"❌ {error_msg}")
        #     log_message(
        #         folder=PIPELINE_MAIN_PROCESS_FOLDER,
        #         raw_manifest_filename=combined_output_filename,
        #         log_string=f"Step 7: Entity Classification - FAILURE. Error: {e}",
        #         level="error"
        #     )
        #     raise

        # Step 11: HS Code Extraction
        try:
            dataframe = extract_hs_code(dataframe, combined_output_filename)
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 11: HS Code Extraction - SUCCESS. Shape: {dataframe.shape}",
                level="info"
            )
            print(f"✅ Step 11: HS Code Extraction processed | 📊 Shape: {dataframe.shape}")
        except Exception as e:
            error_msg = f"Error during Step 11 (HS Code Extraction): {e}"
            print(f"❌ {error_msg}")
            log_message(
                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                raw_manifest_filename=combined_output_filename,
                log_string=f"Step 11: HS Code Extraction - FAILURE. Error: {e}",
                level="error"
            )
            raise

        # Save final processed data
        output_path = CLEANED_TEST_DATA_FOLDER if test_mode else OUTPUT_CLEANED
        os.makedirs(output_path, exist_ok=True)
        final_output_path = os.path.join(output_path, combined_output_filename)
        
        csv_saver(dataframe, output_path, combined_output_filename)
        print(f"✅ Final processed data saved to: {final_output_path}")
        
        log_message(
            folder=PIPELINE_MAIN_PROCESS_FOLDER,
            raw_manifest_filename=combined_output_filename,
            log_string=f"Processing completed successfully. Final shape: {dataframe.shape}",
            level="info"
        )

    except Exception as main_e:
        error_msg = f"Critical error in pipeline: {main_e}"
        print(f"❌ {error_msg}")
        log_message(
            folder=PIPELINE_MAIN_PROCESS_FOLDER,
            raw_manifest_filename=combined_output_filename,
            log_string=error_msg,
            level="error"
        )
        raise

    finally:
        # Move original files to processed folder (only in non-test mode)
        if not test_mode and processed_files:
            try:
                processed_folder = os.path.join(OUTPUT_CLEANED, "processed_originals")
                os.makedirs(processed_folder, exist_ok=True)
                
                for filename in processed_files:
                    try:
                        # Move from processing folder to processed folder
                        source_path = os.path.join(PROCESS_MANIFESTS, filename)
                        dest_path = os.path.join(processed_folder, filename)
                        
                        if os.path.exists(source_path):
                            shutil.move(source_path, dest_path)
                            print(f"✅ Moved {filename} to processed folder")
                            log_message(
                                folder=PIPELINE_MAIN_PROCESS_FOLDER,
                                raw_manifest_filename=filename,
                                log_string=f"File moved to processed folder: {dest_path}",
                                level="info"
                            )
                        else:
                            print(f"⚠️ File {filename} not found in processing folder")
                            
                    except Exception as move_e:
                        print(f"❌ Error moving {filename}: {move_e}")
                        log_message(
                            folder=PIPELINE_MAIN_PROCESS_FOLDER,
                            raw_manifest_filename=filename,
                            log_string=f"Error moving file to processed: {move_e}",
                            level="error"
                        )

                # Clean up processing folder
                if os.path.exists(PROCESS_MANIFESTS):
                    try:
                        shutil.rmtree(PROCESS_MANIFESTS)
                        print(f"✅ Processing folder cleaned up")
                        log_message(
                            folder=PIPELINE_MAIN_PROCESS_FOLDER,
                            raw_manifest_filename="",
                            log_string="Processing folder cleaned up successfully",
                            level="info"
                        )
                    except Exception as cleanup_e:
                        print(f"❌ Error cleaning up processing folder: {cleanup_e}")
                        log_message(
                            folder=PIPELINE_MAIN_PROCESS_FOLDER,
                            raw_manifest_filename="",
                            log_string=f"Error cleaning up processing folder: {cleanup_e}",
                            level="error"
                        )

            except Exception as final_e:
                print(f"❌ Error in final cleanup: {final_e}")
                log_message(
                    folder=PIPELINE_MAIN_PROCESS_FOLDER,
                    raw_manifest_filename="",
                    log_string=f"Error in final cleanup: {final_e}",
                    level="error"
                )

    end_time = time.time()
    total_time = end_time - start_time

    # Convert to hours, minutes, seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n⏱️ Total time taken: {hours}h {minutes}m {seconds:.2f}s")
    log_message(
        folder=PIPELINE_MAIN_PROCESS_FOLDER,
        raw_manifest_filename="",
        log_string=f"Pipeline completed successfully. Total time taken: {hours}h {minutes}m {seconds:.2f}s",
        level="info"
    )
    print(f"🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process manifest files.")
    # Add an argument for test mode
    parser.add_argument("--test", action="store_true", help="Run in test mode using data from Google Sheets.")
    # Parse the arguments
    args = parser.parse_args()

    # Call the pipeline function with the test_mode argument
    pipeline(test_mode=args.test)