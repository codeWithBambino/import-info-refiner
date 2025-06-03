import os
import pandas as pd
import json
import traceback
import time
from tqdm import tqdm
from src.helpers.ai.gemma_handler import GemmaHandler
from src.helpers.logger import log_message
from src.config.folder_name import CITY_EXTRACTION_FOLDER
from src.config.config import CITY_EXTRACTION_PROMPT, TEMP_DIR

# Constants for configuration
CHUNK_SIZE = 20

def process_address_chunk(address_chunk, raw_manifest_filename, country_context="global"):
    """
    Process a chunk of addresses at once using Gemma.
    
    Args:
        address_chunk (list): List of raw addresses to process
        raw_manifest_filename (str): Name of the CSV file for logging purposes
        country_context (str): Country context for extraction ("india", "us", or "global")
        
    Returns:
        dict: Dictionary mapping raw addresses to extracted cities
    """
    # Initialize GemmaHandler
    gemma_handler = GemmaHandler(CITY_EXTRACTION_FOLDER)
    
    # Filter out empty addresses from the chunk before preparing inputs
    valid_addresses_in_chunk = [addr for addr in address_chunk if addr]

    if not valid_addresses_in_chunk:
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string="Skipping empty or invalid address chunk."
        )
        return {}

    # Prepare the inputs for Gemma. Each item in the list is a separate prompt input.
    # The prompt template expects a 'Cities' list with 'raw_address'.
    gemma_inputs = [
        json.dumps({"Cities": [{"raw_address": addr}]}) for addr in valid_addresses_in_chunk
    ]

    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Processing {len(gemma_inputs)} valid addresses with Gemma..."
    )

    # Process the prompts using GemmaHandler's process_prompts for concurrency
    responses = gemma_handler.process_prompts(CITY_EXTRACTION_PROMPT, gemma_inputs)

    # Initialize address map
    address_map = {}

    for i, response in enumerate(responses):
        original_address = valid_addresses_in_chunk[i] # Address sent to Gemma
        extracted_city = None # Default

        if response and response.get('status') == 'success' and response.get('data') and 'Cities' in response['data']:
            cities_data = response['data']['Cities']
            if cities_data: # Expecting one city object per response item
                # The 'raw_address' in Gemma's response should match our 'original_address'
                # And 'city' is the extracted city by Gemma
                city_info = cities_data[0]
                if city_info.get('raw_address') == original_address:
                    extracted_city = city_info.get('city')
                    if extracted_city:
                        log_message(
                            folder=CITY_EXTRACTION_FOLDER,
                            raw_manifest_filename=raw_manifest_filename,
                            log_string=f"Gemma: {original_address} -> {extracted_city}"
                        )
                    else:
                        log_message(
                            folder=CITY_EXTRACTION_FOLDER,
                            raw_manifest_filename=raw_manifest_filename,
                            log_string=f"Gemma response for '{original_address}' missing 'city' field. No city extracted."
                        )
                else:
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Gemma response raw_address mismatch for '{original_address}'. Expected '{original_address}', got '{city_info.get('raw_address')}'. No city extracted."
                    )
            else:
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Gemma response for '{original_address}' missing 'Cities' data. No city extracted."
                )
        else:
            error_msg = response.get('message', 'Unknown error') if isinstance(response, dict) else 'Processing failed'
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Gemma processing failed for '{original_address}': {error_msg}. No city extracted."
            )
        
        if extracted_city: # Only map if we got a city
            address_map[original_address] = extracted_city

    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Successfully mapped {len(address_map)} addresses to cities out of {len(valid_addresses_in_chunk)} processed."
    )
    
    return address_map

def apply_city_extraction(dataframe: pd.DataFrame, address_column: str, city_column: str, 
                         raw_manifest_filename: str, country_context: str = "global",
                         backup_file: str = None) -> pd.DataFrame:
    """
    Extracts city information from a specified address column and adds it to a city column.
    Processes only unique addresses in chunks for efficiency.
    
    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        address_column (str): The column containing addresses to process.
        city_column (str): The column to store extracted cities.
        raw_manifest_filename (str): Name of the CSV file for logging purposes.
        country_context (str): Country context for extraction ("india", "us", or "global").
        backup_file (str): Path to the JSON file for saving backup. If None, uses default.
        
    Returns:
        pd.DataFrame: The DataFrame with added city column.
    """
    # Generate backup filename if not provided
    if backup_file is None:
        backup_file = os.path.join(TEMP_DIR, f"backup_extracted_cities_{address_column.replace(' ', '_')}.json")
    
    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Processing address column: {address_column} -> {city_column}"
    )
    
    # Load existing backup if available
    backup_data = {}
    if os.path.exists(backup_file):
        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Loaded {len(backup_data)} addresses from backup file"
                )
        except json.JSONDecodeError:
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Warning: Could not parse {backup_file}, starting with empty backup."
            )
    
    # Skip if column doesn't exist
    if address_column not in dataframe.columns:
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Column {address_column} not found in DataFrame. Skipping."
        )
        return dataframe
    
    # Get unique addresses from the column (exclude NaN/None and empty strings)
    dataframe_filtered = dataframe[dataframe[address_column].notna() & (dataframe[address_column].str.strip() != '')]
    all_unique_addresses_in_column = dataframe_filtered[address_column].unique().tolist()

    # Filter out addresses that are already in the backup
    # backup_data keys are the raw addresses
    addresses_needing_processing = [addr for addr in all_unique_addresses_in_column if addr not in backup_data]
    
    total_to_process_api = len(addresses_needing_processing)
    total_from_backup = len(all_unique_addresses_in_column) - total_to_process_api

    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Address Column '{address_column}': Total unique valid addresses: {len(all_unique_addresses_in_column)}. "
                   f"To be processed via API: {total_to_process_api}. Loaded from backup: {total_from_backup}."
    )
    
    # Initialize city column if it doesn't exist, otherwise ensure it's suitable
    if city_column not in dataframe.columns:
        dataframe[city_column] = pd.NA # Use pandas NA for missing values for better type handling
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Created new column: {city_column}"
        )
    else:
        # Ensure existing column can store strings or None/NA
        if not pd.api.types.is_string_dtype(dataframe[city_column]) and not pd.api.types.is_object_dtype(dataframe[city_column]):
             dataframe[city_column] = dataframe[city_column].astype(object)

    # Apply backed-up cities first
    if total_from_backup > 0:
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Applying {total_from_backup} cities from backup for address column '{address_column}'..."
        )
        for address in tqdm([addr for addr in all_unique_addresses_in_column if addr in backup_data], desc=f"Applying backup for {address_column}"):
            city = backup_data[address]
            dataframe.loc[dataframe[address_column] == address, city_column] = city
            
    # Process remaining unique addresses in chunks with progress bar
    if addresses_needing_processing:
        progress_bar = tqdm(range(0, total_to_process_api, CHUNK_SIZE), desc=f"Extracting cities for {address_column} via API")
        for chunk_start in progress_bar:
            chunk_end = min(chunk_start + CHUNK_SIZE, total_to_process_api)
            current_chunk_to_process = addresses_needing_processing[chunk_start:chunk_end]
            
            if not current_chunk_to_process:
                continue

            progress_bar.set_postfix_str(f"API Processing: {chunk_start}-{chunk_end} of {total_to_process_api}")
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Processing chunk {chunk_start}-{chunk_end} for '{address_column}' via API ({len(current_chunk_to_process)} addresses)"
            )
            try:
                address_map = process_address_chunk(current_chunk_to_process, raw_manifest_filename, country_context)
                
                if not address_map:
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string="Warning: No addresses were mapped to cities in this API chunk!"
                    )
                else:
                    updated_rows_in_chunk = 0
                    for address, city in address_map.items():
                        # Update all rows with this address
                        dataframe.loc[dataframe[address_column] == address, city_column] = city
                        updated_rows_in_chunk += dataframe[dataframe[address_column] == address].shape[0]
                        # Add to backup
                        backup_data[address] = city
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Updated {updated_rows_in_chunk} DataFrame rows from API results for this chunk."
                    )
                
                # Save progress to backup file after each chunk
                try:
                    with open(backup_file, "w") as f:
                        json.dump(backup_data, f, indent=2)
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Saved {len(backup_data)} addresses to backup file: {backup_file}"
                    )
                except IOError as e:
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Error saving backup file {backup_file}: {e}"
                    )

            except Exception as e:
                error_msg = f"Error processing API chunk {chunk_start}-{chunk_end} for '{address_column}': {str(e)}"
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"{error_msg}\n{traceback.format_exc()}"
                )
            
            # Add a small delay between chunks to avoid rate limiting
            if chunk_end < total_to_process_api: # Avoid sleep after the last chunk
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string="Waiting for 3 seconds before processing next API chunk..."
                )
                time.sleep(3)
    
    # Check how many cities were extracted
    city_count = dataframe[dataframe[city_column].notna()].shape[0]
    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Extraction complete. Added city data for {city_count} out of {len(dataframe)} rows ({city_count/len(dataframe)*100:.2f}%)"
    )
    
    return dataframe