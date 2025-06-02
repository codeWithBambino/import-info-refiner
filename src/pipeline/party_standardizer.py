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

CHUNK_SIZE = 10

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
    
    # Prepare the input for Gemma as a JSON batch
    company_batch = {
        "Companies": [{
            "Raw Name": name
        } for name in cleaned_raw_names]
    }
    
    # Process the prompt using GemmaHandler
    response = gemma_handler.process_prompt(PARTY_STANDARDIZER_PROMPT, company_batch)
    status = response['status']
    
    
    # Create a mapping between raw names and cleaned names
    name_map = {}
    
    if status and response and response.get('data') and 'Companies' in response['data']:
        companies_data = response['data']['Companies']
        
        # Map raw names to their cleaned versions
        for raw_name, cleaned_raw in zip(name_chunk, cleaned_raw_names):
            cleaned_name = None
            
            # Find matching company in response
            for item in companies_data:
                if item['Raw Name'] == cleaned_raw:
                    cleaned_name = item['Cleaned']
                    break
            
            if cleaned_name is None:
                # Fall back to the cleaned raw name if no match found
                cleaned_name = cleaned_raw
            
            # Store in map using raw name as key
            name_map[raw_name] = cleaned_name
            
            # Log the standardization
            log_message(
                folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"{raw_name} -> {cleaned_name}"
            )
    else:
        # If processing failed, create map using cleaned raw names
        error_msg = response.get('message', 'Unknown error') if isinstance(response, dict) else 'Processing failed'
        log_message(
            folder=STANDARDIZE_PARTY_NAMES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Gemma processing failed: {error_msg}"
        )
        
        for raw_name, cleaned_raw in zip(name_chunk, cleaned_raw_names):
            name_map[raw_name] = cleaned_raw
            log_message(
                folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"{raw_name} -> {cleaned_raw} (fallback)"
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
        unique_names = dataframe[column].unique().tolist()
        total_unique = len(unique_names)
        log_message(
            folder=STANDARDIZE_PARTY_NAMES_FOLDER,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Found {total_unique} unique names out of {len(dataframe)} total records"
        )
        
        # Create a new column for cleaned names
        cleaned_column = f"cleaned_{column}"
        dataframe[cleaned_column] = dataframe[column]
        
        # Process unique names in chunks with progress bar
        progress_bar = tqdm(range(0, total_unique, CHUNK_SIZE), desc=f"Standardizing {column}")
        for chunk_start in progress_bar:
            chunk_end = min(chunk_start + CHUNK_SIZE, total_unique)
            current_chunk = unique_names[chunk_start:chunk_end]
            
            # Skip processing if all names in this chunk are already in the backup
            already_processed = all(name in backup_data.get(column, {}) for name in current_chunk)
            
            if already_processed:
                # Load from backup and map to all occurrences
                progress_bar.set_postfix_str(f"Loading from backup: {chunk_start}-{chunk_end}")
                log_message(
                    folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Loading chunk {chunk_start}-{chunk_end} from backup"
                )
                for name in current_chunk:
                    cleaned_name = backup_data[column][name]
                    # Update all rows with this raw name
                    dataframe.loc[dataframe[column] == name, cleaned_column] = cleaned_name
            else:
                # Process the chunk
                progress_bar.set_postfix_str(f"Processing: {chunk_start}-{chunk_end}")
                log_message(
                    folder=STANDARDIZE_PARTY_NAMES_FOLDER,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=f"Processing chunk {chunk_start}-{chunk_end}"
                )
                name_map = process_chunk(current_chunk, raw_manifest_filename)
                
                # Update DataFrame and backup
                for raw_name, cleaned_name in name_map.items():
                    # Update all rows with this raw name
                    dataframe.loc[dataframe[column] == raw_name, cleaned_column] = cleaned_name
                    
                    # Add to backup
                    if column not in backup_data:
                        backup_data[column] = {}
                    backup_data[column][raw_name] = cleaned_name
                
                # Save progress to backup file after each chunk
                with open(backup_file, "w") as f:
                    json.dump(backup_data, f, indent=2)
                
                # Add a small delay between chunks to avoid rate limiting
                time.sleep(3)
        
        # Replace original column with cleaned version
        dataframe.drop(columns=column, inplace=True)
        dataframe.rename(columns={cleaned_column: column}, inplace=True)
    
    return dataframe