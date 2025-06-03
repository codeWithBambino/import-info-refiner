import os
import pandas as pd
import json
import time
from tqdm import tqdm
from src.helpers.ai.gemma_handler import GemmaHandler
from src.helpers.logger import log_message
from src.config.folder_name import STANDARDIZE_PARTY_NAMES_FOLDER
from src.config.config import PARTY_STANDARDIZER_PROMPT, TEMP_DIR, COLUMNS_TO_STANDARDIZE
from src.config.abbreviations import ABBREVIATIONS, UNWANTED_TOKENS
import re

CHUNK_SIZE = 20

def clean_party_name(name: str, raw_manifest_filename: str) -> str:

    """
    Cleans a single party name using a robust and ordered series of transformations.
    Optimized for correcting messy, inconsistent formats commonly seen in party names.

    Args:
        name (str): The raw company/party name.

    Returns:
        str: The cleaned and standardized name.
    """
    try:
        # Step 1: Initial whitespace normalization (foundational)
        cleaned = re.sub(r'[\s\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]+', ' ', name)  # Normalize all whitespace types
        cleaned = cleaned.strip()  # Remove leading/trailing spaces
        
        # Step 2: Convert to uppercase (must happen early but after whitespace normalization)
        cleaned = cleaned.upper()  # Unify case for consistent pattern matching
        
        # Step 3: Extract content from parentheses (before removing special characters)
        cleaned = re.sub(r'\(([^)]+)\)', r'\1', cleaned)  # Preserve parenthetical content
        
        # Step 4: Fix dot-separated initials (before removing punctuation)
        cleaned = re.sub(r'\b([A-Z])\.([A-Z])\.', r'\1 \2 ', cleaned)  # Handle double initials
        cleaned = re.sub(r'\b([A-Z])\.([A-Z])', r'\1 \2', cleaned)  # Handle paired initials
        cleaned = re.sub(r'\b([A-Z])\.', r'\1', cleaned)  # Handle single initials
        
        # Step 5: Normalize punctuation by replacing hyphens and periods with spaces
        cleaned = re.sub(r'[-.]', ' ', cleaned)  # Convert punctuation to spaces
        
        # Step 6: Remove special characters (after handling specific punctuation)
        cleaned = re.sub(r'[^A-Z0-9\s,]', ' ', cleaned)  # Remove special chars except commas
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Clean up resulting whitespace
        
        # Step 7: Remove unwanted tokens (after case conversion, before company type normalization)
        for token in UNWANTED_TOKENS:
            cleaned = re.sub(r'\b' + re.escape(token) + r'\b', '', cleaned)  # Remove standard tokens with word boundaries
        
        # Step 8: Remove leading numbers (early cleanup before company name standardization)
        cleaned = re.sub(r'^\d+\s+', '', cleaned)  # Strip leading numeric identifiers
        
        # Step 9: Remove location indicators (before company suffix standardization)
        cleaned = re.sub(r'\b(IN|I|IND|INDI|INDIA)\b', '', cleaned)  # Remove country references
        
        # Step 10: Apply custom abbreviation replacements (sort by length to handle longer patterns first)
        sorted_abbrevs = sorted(ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True)
        for abbr, full_form in sorted_abbrevs:
            # Use word boundaries to ensure we're matching whole words
            pattern = r'\b' + re.escape(abbr) + r'\b'
            cleaned = re.sub(pattern, full_form, cleaned)
        
        # Step 11: Standardize company type variations (more specific patterns first)
        cleaned = re.sub(r'\b(PRIVATELT|PRIVA|PRIVATELTD|PRIVATELIM|PRIVATELIMITED|PVT\.?\s*LT|PVT\.?\s*LTD|P\.?\s*LTD)\b', 
                        'PRIVATE LIMITED', cleaned)  # Standardize private limited variants
        cleaned = re.sub(r'\b(LIMI|LI|L|LIMIT|LIM|LIMITE)\b', 'LIMITED', cleaned)  # Standardize limited variants
        cleaned = re.sub(r'\b(P|PVT|OP|CO)\b(?!\s*LIMITED)', 'PRIVATE', cleaned)  # Handle abbreviations
        
        # Step 12: Remove duplicate company type suffixes (after standardization)
        cleaned = re.sub(r'\bPRIVATE\s+LIMITED\s+PRIVATE\s+LIMITED\b', 'PRIVATE LIMITED', cleaned)  # Fix duplicated suffixes
        cleaned = re.sub(r'\bPRIVATE\s+PRIVATE\b', 'PRIVATE', cleaned)  # Fix duplicated terms
        
        # Step 13: Fix misplaced commas before suffixes (after company type standardization)
        cleaned = re.sub(r',\s+(INC|LTD|LLC|LLP|CORP)\b', r' \1', cleaned)  # Fix comma placement
        
        # Step 14: Remove department or branch descriptors (near end of cleaning)
        cleaned = re.sub(r'\s+(HQ|BRANCH|DIVISION|UNIT|PLANT|SECTION|CENTER|CENTRE)$', '', cleaned)  # Remove dept indicators
        
        # Step 15: Remove trailing code-like tokens (final specific cleanup)
        cleaned = re.sub(r'(?<!\,)\s+[A-Z]\s+[A-Z0-9-]*\d[A-Z0-9-]*$', '', cleaned)  # Remove trailing codes
        
        # Step 16: Final whitespace cleanup (always last)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Ensure single spaces throughout

        return cleaned

    except Exception as e:
        
        log_message(
            folder=STANDARDIZE_PARTY_NAMES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Exception while cleaning party name: {e}\nName: {name}"
        )
        return name

def process_chunk(name_chunk, raw_manifest_filename):
    """
    Process a chunk of company names at once using Gemma.
    
    Args:
        name_chunk (list): List of raw company names to process
        raw_manifest_filename (str): Name of the CSV file for logging purposes
        
    Returns:
        dict: Dictionary mapping raw names to cleaned names
    """
    # Initialize GemmaHandler
    gemma_handler = GemmaHandler(STANDARDIZE_PARTY_NAMES_FOLDER)
    
    # Clean each name in the chunk first
    cleaned_raw_names = [clean_party_name(name, raw_manifest_filename) for name in name_chunk]
    
    # Prepare the inputs for Gemma as a list of JSON-like dicts for batching
    # Each item in the list will be a separate prompt input for Gemma
    # The prompt template itself expects a 'Companies' list with 'Raw Name'
    gemma_inputs = [
        json.dumps({"Companies": [{"Raw Name": name}]}) for name in cleaned_raw_names
    ]

    # Process the prompts using GemmaHandler's process_prompts for concurrency
    responses = gemma_handler.process_prompts(PARTY_STANDARDIZER_PROMPT, gemma_inputs)

    # Create a mapping between raw names and cleaned names
    name_map = {}

    for i, response in enumerate(responses):
        raw_name_original = name_chunk[i]  # Original name from the input chunk
        cleaned_raw_name = cleaned_raw_names[i] # Pre-cleaned name sent to Gemma
        final_cleaned_name = cleaned_raw_name # Default to pre-cleaned name

        if response and response.get('status') == 'success' and response.get('data') and 'Companies' in response['data']:
            companies_data = response['data']['Companies']
            if companies_data: # Expecting one company per response item
                # The 'Raw Name' in Gemma's response should match our 'cleaned_raw_name'
                # And 'Cleaned' is the name standardized by Gemma
                gemma_standardized_name = companies_data[0].get('Cleaned')
                if gemma_standardized_name:
                    final_cleaned_name = gemma_standardized_name
                else:
                    log_message(
                        folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Gemma response for '{cleaned_raw_name}' missing 'Cleaned' field. Using pre-cleaned name."
                    )
            else:
                log_message(
                    folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Gemma response for '{cleaned_raw_name}' missing 'Companies' data. Using pre-cleaned name."
                )
        else:
            error_msg = response.get('message', 'Unknown error') if isinstance(response, dict) else 'Processing failed'
            log_message(
                folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Gemma processing failed for '{cleaned_raw_name}': {error_msg}. Using pre-cleaned name."
            )
        
        name_map[raw_name_original] = final_cleaned_name
        log_message(
            folder=STANDARDIZE_PARTY_NAMES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"{raw_name_original} -> {final_cleaned_name}"
        )
    
    return name_map

def party_standardizer(dataframe: pd.DataFrame, raw_manifest_filename: str, backup_file: str = "backup_cleaned_party_names.json") -> pd.DataFrame:
    """
    Standardizes the party names in the specified columns of the DataFrame.
    Processes only unique names in chunks for efficiency.
    Backs up progress to a JSON file after each processed chunk.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        raw_manifest_filename (str): Name of the CSV file for logging purposes.
        backup_file (str): Path to the JSON file for saving backup.

    Returns:
        pd.DataFrame: The DataFrame with standardized party names.
    """
    # Load existing backup if available
    backup_data = {}
    backup_file = os.path.join(TEMP_DIR, backup_file)
    if os.path.exists(backup_file):
        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)
        except json.JSONDecodeError:
            log_message(
                folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Warning: Could not parse {backup_file}, starting with empty backup."
            )

    for column in COLUMNS_TO_STANDARDIZE:
        log_message(
            folder=STANDARDIZE_PARTY_NAMES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Processing column: {column}"
        )
        
        # Get unique names from the column
        all_unique_names_in_column = dataframe[column].unique().tolist()
        
        # Filter out names that are already in the backup for this column
        names_to_process_from_backup = backup_data.get(column, {})
        names_needing_processing = [name for name in all_unique_names_in_column if name not in names_to_process_from_backup]
        
        total_to_process_api = len(names_needing_processing)
        total_from_backup = len(all_unique_names_in_column) - total_to_process_api

        log_message(
            folder=STANDARDIZE_PARTY_NAMES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Column '{column}': Total unique names: {len(all_unique_names_in_column)}. "
                       f"To be processed via API: {total_to_process_api}. Loaded from backup: {total_from_backup}."
        )
        
        # Create a new column for cleaned names
        cleaned_column = f"cleaned_{column}"
        dataframe[cleaned_column] = dataframe[column] # Initialize with original names

        # Apply backed-up names first
        if names_to_process_from_backup:
            log_message(
                folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"Applying {len(names_to_process_from_backup)} names from backup for column '{column}'..."
            )
            for name, cleaned_name in tqdm(names_to_process_from_backup.items(), desc=f"Applying backup for {column}"):
                dataframe.loc[dataframe[column] == name, cleaned_column] = cleaned_name
        
        # Process remaining unique names in chunks with progress bar
        if names_needing_processing:
            progress_bar = tqdm(range(0, total_to_process_api, CHUNK_SIZE), desc=f"Standardizing {column} via API")
            for chunk_start in progress_bar:
                chunk_end = min(chunk_start + CHUNK_SIZE, total_to_process_api)
                current_chunk_to_process = names_needing_processing[chunk_start:chunk_end]
                
                if not current_chunk_to_process: # Should not happen if names_needing_processing is not empty
                    continue

                progress_bar.set_postfix_str(f"API Processing: {chunk_start}-{chunk_end} of {total_to_process_api}")
                log_message(
                    folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Processing chunk {chunk_start}-{chunk_end} for column '{column}' via API ({len(current_chunk_to_process)} names)"
                )
                name_map = process_chunk(current_chunk_to_process, raw_manifest_filename)
                
                # Update DataFrame and backup
                for raw_name, cleaned_name in name_map.items():
                    # Update all rows with this raw name
                    dataframe.loc[dataframe[column] == raw_name, cleaned_column] = cleaned_name
                    
                    # Add to backup
                    if column not in backup_data:
                        backup_data[column] = {}
                    backup_data[column][raw_name] = cleaned_name
                
                # Save progress to backup file after each chunk
                try:
                    with open(backup_file, "w") as f:
                        json.dump(backup_data, f, indent=2)
                except IOError as e:
                    log_message(
                        folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Error saving backup file {backup_file}: {e}"
                    )
                
                # Add a small delay between chunks to avoid rate limiting
                if chunk_end < total_to_process_api: # Avoid sleep after the last chunk
                    time.sleep(3)
        
        # Replace original column with cleaned version
        dataframe.drop(columns=column, inplace=True)
        dataframe.rename(columns={cleaned_column: column}, inplace=True)
    
    return dataframe