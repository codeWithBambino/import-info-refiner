import pandas as pd
from tqdm import tqdm
import os, json
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
    with open(ENTITY_CLASSIFICATION_PROMPT, "r") as file:
        prompt_template = file.read()
    prompt = prompt_template.replace("{gemini_custom_input}", f"{chunk}")
    return prompt


def classify_entities(dataframe: pd.DataFrame, raw_manifest_filename: str, manual_mode:bool = True ) -> pd.DataFrame:
    """
    Classify entities in a DataFrame using the Gemini model.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the entities to classify.
        raw_manifest_filename (str): Name of the original file for logging purposes.

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

            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, f"🔍 Classifying column: {column} ({len(unique_company_list)} unique entities)", level="info")

            if manual_mode:
                manual_data = {}
                for file_name in os.listdir(MANUAL_CLASSIFICATION_FOLDER_PATH):
                    if file_name.endswith(".json"):
                        file_path = os.path.join(MANUAL_CLASSIFICATION_FOLDER_PATH, file_name)
                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                manual_data.update(data)
                            except json.JSONDecodeError:
                                print(f"[ERROR] Skipping invalid JSON: {file_name}")

                results = manual_data  # or however you intend to use this data
                print(f"{column}: Manual data loaded successfully.")
            else:
                for chunk_number, chunk in tqdm(chunked_list.items(), desc=f"⏳ Processing '{column}'", unit="chunk"):
                    try:
                        prompt = generate_prompt(chunk)

                        # Log the prompt
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                    f"📤 Prompt for chunk {chunk_number}:\n{prompt}", level="info")

                        response = ask_gemini(prompt)

                        # Log the response
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename,
                                    f"📥 Response for chunk {chunk_number}:\n{response}", level="info")

                        if isinstance(response, dict):
                            results.update(response)
                        else:
                            warning_msg = f"⚠️ Unexpected response format in chunk {chunk_number}:\n{response}"
                            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, warning_msg, level="warning")

                    except Exception as chunk_err:
                        error_msg = f"❌ Error in chunk {chunk_number}: {chunk_err}"
                        log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, error_msg, level="error")

            dataframe = add_category_column(dataframe, column, results)

        except Exception as col_err:
            error_msg = f"🚨 Failed to process column '{column}': {col_err}"
            log_message(ENTITY_CLASSIFICATION_FOLDER, raw_manifest_filename, error_msg, level="error")

    return dataframe