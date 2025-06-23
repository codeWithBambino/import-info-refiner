import pandas as pd
from tqdm import tqdm
import os, json
import ast
from src.helpers.ai.gemini_handler import ask_gemini
from src.config.config import (
    ENTITY_CLASSIFICATION_PROMPT,
    ENTITY_COLUMNS_TO_CLASSIFY,
    ENTITY_CHUNK_SIZE,
    MANUAL_CLASSIFICATION_FOLDER_PATH
)
from src.helpers.classifier_extractor import add_category_column
from src.helpers.logger import log_message
from src.config.folder_name import ENTITY_CLASSIFICATION_FOLDER


def chunk_to_dict(company_list):
    return {
        i: company_list[i:i + ENTITY_CHUNK_SIZE]
        for i in range(0, len(company_list), ENTITY_CHUNK_SIZE)
    }


def generate_prompt(chunk):
    """
    Generate prompt with properly formatted JSON input for Gemini
    """
    with open(ENTITY_CLASSIFICATION_PROMPT, "r") as file:
        prompt_template = file.read()
    
    # Format the company list as the expected JSON structure
    companies_json = {
        "companies": [
            {"company_name": company} for company in chunk
        ]
    }
    
    # Convert to JSON string for the prompt
    companies_json_str = json.dumps(companies_json, indent=2)
    
    # Replace placeholder with formatted JSON
    prompt = prompt_template.replace("{gemini_custom_input}", companies_json_str)
    return prompt


def classify_entities(dataframe: pd.DataFrame, raw_manifest_filename: str, manual_mode: bool = True) -> pd.DataFrame:
    """
    Classify entities in a DataFrame using the Gemini model.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the entities to classify.
        raw_manifest_filename (str): Name of the original file for logging purposes.
        manual_mode (bool): Whether to use manual classification mode.

    Returns:
        pd.DataFrame: The DataFrame with additional category columns.
    """

    for column in ENTITY_COLUMNS_TO_CLASSIFY:
        try:
            if column not in dataframe.columns:
                error_msg = f"⚠️ Column '{column}' not found in DataFrame. Skipping."
                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, error_msg, level="warning")
                continue

            unique_company_list = dataframe[column].dropna().unique().tolist()

            if not unique_company_list:
                info_msg = f"🛑 No data to classify in column '{column}'. Skipping."
                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, info_msg, level="info")
                continue

            chunked_list = chunk_to_dict(unique_company_list)
            results = {}

            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, 
                       f"🔍 Classifying column: {column} ({len(unique_company_list)} unique entities)", level="info")

            if manual_mode:
                # Load manual classification data
                manual_data = {}
                for file_name in os.listdir(MANUAL_CLASSIFICATION_FOLDER_PATH):
                    if file_name.endswith(".json"):
                        file_path = os.path.join(MANUAL_CLASSIFICATION_FOLDER_PATH, file_name)
                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                manual_data.update(data)
                            except json.JSONDecodeError:
                                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                           f"[ERROR] Skipping invalid JSON: {file_name}", level="error")

                results = manual_data
                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                           f"{column}: Manual data loaded successfully.", level="info")
            else:
                # Step 1: Load all manual data once
                manual_data = {}
                if os.path.exists(MANUAL_CLASSIFICATION_FOLDER_PATH):
                    for file_name in os.listdir(MANUAL_CLASSIFICATION_FOLDER_PATH):
                        if file_name.endswith(".json"):
                            file_path = os.path.join(MANUAL_CLASSIFICATION_FOLDER_PATH, file_name)
                            try:
                                with open(file_path, "r") as f:
                                    data = json.load(f)
                                    if "categorized_data" in data:
                                        for entry in data["categorized_data"]:
                                            name = entry.get("company_name")
                                            category = entry.get("category")
                                            if name and category:
                                                manual_data[name.strip().upper()] = category
                            except Exception as e:
                                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                           f"⚠️ Failed to read manual classification file {file_name}: {e}", level="warning")

                for chunk_number, chunk in tqdm(chunked_list.items(), desc=f"⏳ Processing '{column}'", unit="chunk"):
                    try:
                        # Step 2: Split into known and unknown companies
                        chunk_set = set([c.strip().upper() for c in chunk])
                        known_chunk_results = {
                            # Use original company name as key, not uppercase
                            next(name for name in chunk if name.strip().upper() == upper_name): manual_data[upper_name] 
                            for upper_name in chunk_set if upper_name in manual_data
                        }
                        unknown_chunk = [name for name in chunk if name.strip().upper() not in manual_data]

                        # Update results with known values first
                        results.update(known_chunk_results)

                        # If nothing left to classify, skip Gemini call
                        if not unknown_chunk:
                            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                       f"✅ Chunk {chunk_number} fully classified from manual data. Skipping Gemini.", level="info")
                            continue

                        # Log new companies that need classification
                        if unknown_chunk:
                            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                       f"🆕 New companies needing Gemini classification in chunk {chunk_number}: {unknown_chunk}", level="info")

                        # Generate prompt with proper JSON formatting
                        prompt = generate_prompt(unknown_chunk)

                        # Log the prompt (truncated for readability)
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                   f"📤 Prompt for chunk {chunk_number} (first 500 chars):\n{prompt[:500]}...", level="info")

                        # Call Gemini
                        response = ask_gemini(prompt)

                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                    f"📨 FULL Gemini response for chunk {chunk_number}:\n{response}", level="debug")

                        # Parse response
                        parsed_response = None
                        if isinstance(response, str):
                            response_clean = response.strip().replace("```json", "").replace("```", "").strip()
                            try:
                                parsed_response = json.loads(response_clean)
                            except json.JSONDecodeError:
                                try:
                                    parsed_response = ast.literal_eval(response_clean)
                                except Exception as e:
                                    log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                               f"❌ Both JSON and literal_eval failed in chunk {chunk_number}: {e}", level="error")
                                    log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                               f"Raw response: {response_clean[:1000]}", level="error")
                                    continue
                        elif isinstance(response, dict):
                            parsed_response = response
                        else:
                            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                       f"❌ Unexpected response type in chunk {chunk_number}: {type(response)}", level="error")
                            continue

                        # Debug log for parsed response
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                   f"✅ Parsed response type: {type(parsed_response)} | Keys: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else 'Not a dict'}", level="info")

                        # Extract categorized data - handle both possible response formats
                        categorized_data = None
                        if isinstance(parsed_response, dict):
                            categorized_data = parsed_response.get("categorized_data")
                        if categorized_data is None and isinstance(parsed_response, list):
                            categorized_data = parsed_response

                        if categorized_data and isinstance(categorized_data, list):
                            for entry in categorized_data:
                                if isinstance(entry, dict):
                                    name = entry.get("company_name")
                                    category = entry.get("category")
                                    if name and category:
                                        results[name] = category
                                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                                   f"✅ Classified: {name} -> {category}", level="info")
                        else:
                            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                       f"⚠️ Could not extract categorized_data from response in chunk {chunk_number}. Response structure: {parsed_response}", level="warning")

                    except Exception as chunk_err:
                        error_msg = f"❌ Error in chunk {chunk_number}: {str(chunk_err)}"
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, error_msg, level="error")
                        import traceback
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, 
                                   f"Full traceback: {traceback.format_exc()}", level="error")

            # Add category column to dataframe directly
            category_column_name = f"{column} Category"
            
            # Debug: Log results structure
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                       f"🔍 Results dict has {len(results)} entries. Sample: {dict(list(results.items())[:3])}", level="info")
            
            # Debug: Log unique values in dataframe column
            unique_df_values = dataframe[column].dropna().unique()[:5]  # First 5 for debugging
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                       f"🔍 DataFrame {column} sample values: {list(unique_df_values)}", level="info")
            
            # Create normalized lookup for case-insensitive matching
            normalized_results = {}
            for company_name, category in results.items():
                # Normalize key: strip whitespace and convert to uppercase
                normalized_key = str(company_name).strip().upper()
                normalized_results[normalized_key] = category
            
            if len(results) < len(unique_company_list):
                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                            f"❗ Warning: Only {len(results)} out of {len(unique_company_list)} classified. Possible Gemini response issue.", level="warning")
            
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                       f"🔍 Normalized results sample: {dict(list(normalized_results.items())[:3])}", level="info")
            
            # Create a mapping function with better matching
            def get_category(company_name):
                if pd.isna(company_name):
                    return "Unknown"
                
                # First try exact match
                if company_name in results:
                    return results[company_name]
                
                # Then try normalized match
                normalized_name = str(company_name).strip().upper()
                if normalized_name in normalized_results:
                    return normalized_results[normalized_name]
                
                # Log unmatched companies for debugging
                log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                           f"⚠️ No match found for: '{company_name}' (normalized: '{normalized_name}')", level="warning")
                return "Unclassified"
            
            # Apply the mapping to create the new column
            dataframe[category_column_name] = dataframe[column].apply(get_category)
            
            # Count the results
            category_counts = dataframe[category_column_name].value_counts()
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                       f"✅ Added column '{category_column_name}'. Distribution: {category_counts.to_dict()}", level="info")
            
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                       f"✅ Successfully processed column '{column}' with {len(results)} classifications", level="info")

        except Exception as col_err:
            error_msg = f"🚨 Failed to process column '{column}': {str(col_err)}"
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, error_msg, level="error")
            import traceback
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, 
                       f"Full traceback: {traceback.format_exc()}", level="error")

    return dataframe