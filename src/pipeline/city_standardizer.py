import os
import pandas as pd
import json
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
    
    # Prepare the input for Gemma as a JSON batch
    address_batch = {"Cities": [{"raw_address": addr} for addr in address_chunk if addr]}
    
    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Processing {len(address_batch['Cities'])} addresses..."
    )
    
    # Process the prompt using GemmaHandler
    response = gemma_handler.process_prompt(CITY_EXTRACTION_PROMPT, address_batch)
    status = response['status']
    
    # Initialize address map
    address_map = {}
    
    if status and response and response.get('data') and 'Cities' in response['data']:
        cities_data = response['data']['Cities']
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Found {len(cities_data)} cities in the extracted data"
        )
        
        # Map raw addresses to their extracted cities using zip
        for raw_addr, city_data in zip(address_chunk, cities_data):
            if not raw_addr:  # Skip empty addresses
                continue
                
            city = city_data.get('city')
            if city:  # Only map if we got a city
                address_map[raw_addr] = city
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Gemma: {raw_addr} -> {city}"
                )
        
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Successfully mapped {len(address_map)} addresses to cities"
        )
    else:
        error_msg = response.get('message', 'Unknown error') if isinstance(response, dict) else 'Processing failed'
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Error processing addresses: {error_msg}"
        )
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Gemma processing failed: {error_msg}"
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
    
    # Get unique addresses from the column (exclude NaN/None)
    dataframe_filtered = dataframe[dataframe[address_column].notna()]
    unique_addresses = dataframe_filtered[address_column].unique().tolist()
    total_unique = len(unique_addresses)
    log_message(
        folder=CITY_EXTRACTION_FOLDER,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Found {total_unique} unique addresses out of {len(dataframe)} total records"
    )
    
    # Initialize city column if it doesn't exist
    if city_column not in dataframe.columns:
        dataframe[city_column] = None
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Created new column: {city_column}"
        )
    
    # Process unique addresses in chunks with progress bar
    progress_bar = tqdm(range(0, total_unique, CHUNK_SIZE), desc="Processing address chunks")
    for chunk_start in progress_bar:
        chunk_end = min(chunk_start + CHUNK_SIZE, total_unique)
        current_chunk = unique_addresses[chunk_start:chunk_end]
        progress_bar.set_postfix({"chunk_size": len(current_chunk)})
        log_message(
            folder=CITY_EXTRACTION_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Processing chunk {chunk_start}-{chunk_end} ({len(current_chunk)} addresses)"
        )
        
        # Skip processing if all addresses in this chunk are already in the backup
        non_empty_addresses = [addr for addr in current_chunk if addr]
        if non_empty_addresses:
            already_processed = all(addr in backup_data for addr in non_empty_addresses)
        else:
            already_processed = True
        
        if already_processed:
            # Load from backup and map to all occurrences
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Loading chunk {chunk_start}-{chunk_end} from backup"
            )
            updated_count = 0
            for address in current_chunk:
                if not address:
                    continue
                
                if address in backup_data:
                    city = backup_data[address]
                    # Update all rows with this raw address
                    dataframe.loc[dataframe[address_column] == address, city_column] = city
                    updated_count += 1
                else:
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Warning: Address not found in backup: {address[:50]}..."
                    )
            
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Updated {updated_count} rows from backup"
            )
        else:
            # Process the chunk
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Processing chunk {chunk_start}-{chunk_end}"
            )
            try:
                address_map = process_address_chunk(current_chunk, raw_manifest_filename, country_context)
                
                if not address_map:
                    log_message(
                        folder=CITY_EXTRACTION_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string="Warning: No addresses were mapped to cities in this chunk!"
                    )
                    continue
                
                # Update DataFrame and backup
                updated_count = 0
                for address, city in address_map.items():
                    # Update all rows with this address
                    match_count = dataframe[dataframe[address_column] == address].shape[0]
                    dataframe.loc[dataframe[address_column] == address, city_column] = city
                    updated_count += match_count
                    
                    # Add to backup
                    backup_data[address] = city
                
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Updated {updated_count} rows with extracted cities"
                )
                
                # Save progress to backup file after each chunk
                with open(backup_file, "w") as f:
                    json.dump(backup_data, f, indent=2)
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Saved {len(backup_data)} addresses to backup file"
                )
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_start}-{chunk_end}: {str(e)}"
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"{error_msg}\n{traceback.format_exc()}"
                )
                # Log the error
                log_message(
                    folder=CITY_EXTRACTION_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=error_msg
                )
            
            # Add a small delay between chunks to avoid rate limiting
            log_message(
                folder=CITY_EXTRACTION_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string="Waiting for 3 seconds before processing next chunk..."
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