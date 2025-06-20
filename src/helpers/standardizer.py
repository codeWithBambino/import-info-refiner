import os
import pandas as pd
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.helpers.logger import log_message

from src.helpers.ai.gemma_handler import GemmaHandler
from src.helpers.logger import log_message
from src.config.config import (
    TEMP_DIR,
    GEMMA_NUM_THREADS,   # fixed cap from config
    DATA_CHUNK_SIZE,     # e.g. 10, defined in config.py
)
from src.config.abbreviations import ABBREVIATIONS, UNWANTED_TOKENS


def local_clean_name(name: str) -> str:
    """
    Perform basic regex‐and abbreviation‐based cleanup on a raw party name.
    """
    # No logging here since this is a simple utility
    if pd.isna(name):
        return ""
    # 1. Normalize to uppercase and strip whitespace
    cleaned = name.strip().upper()
    # 2. Replace any punctuation (non‐alphanumeric, non‐whitespace) with a space
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    # 3. Collapse multiple spaces into one
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # 4. Tokenize
    tokens = cleaned.split()
    # 5. Remove tokens listed in UNWANTED_TOKENS
    tokens = [t for t in tokens if t not in UNWANTED_TOKENS]
    # 6. Replace any token found in ABBREVIATIONS
    tokens = [ABBREVIATIONS.get(t, t) for t in tokens]
    # 7. Rejoin into a single string
    return " ".join(tokens)


def process_batch(
    batch_values: list[str],
    column: str,
    batch_index: int,
    raw_manifest_filename: str,
    STANDARDIZER_PROMPT: str,
    FOLDER_NAME: str,
    city_flag:bool
):
    """
    Process a list of up to DATA_CHUNK_SIZE pre-cleaned names for 'column'.
    - Build a single JSON payload: {"standardized_data":[{"raw_input": name1}, ...]}
    - Send to Gemma; extract each "output" value from response_data["standardized_data"][i]["output"].
    - Retries if Gemma errors or returns invalid JSON (up to 3 attempts).
    - Persist partial results to temp_<column>_cleaned_batch_<batch_index>.json so we can resume.
    """
    folder = FOLDER_NAME
    log_message(
        folder=folder,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Starting processing batch {batch_index} for column '{column}' with {len(batch_values)} names.",
        level="info",
    )

    temp_folder = os.path.join(TEMP_DIR, folder)
    os.makedirs(temp_folder, exist_ok=True)

    temp_file = os.path.join(
        temp_folder, f"temp_{column}_cleaned_batch_{batch_index}.json"
    )

    # 1) Load existing mapping if present (resume capability)
    if os.path.exists(temp_file):
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                batch_mapping = json.load(f)

            log_message(
                folder=folder,
                raw_manifest_filename=raw_manifest_filename,
                log_string=(
                    f"Loaded existing temp file for batch {batch_index}, column '{column}', "
                    f"entries: {len(batch_mapping)}."
                ),
                level="info",
            )
        except Exception as e:
            log_message(
                folder=folder,
                raw_manifest_filename=raw_manifest_filename,
                log_string=(
                    f"Error reading existing temp file for batch {batch_index}, "
                    f"column '{column}': {e}. Starting fresh."
                ),
                level="error",
            )
            batch_mapping = {}
    else:
        batch_mapping = {}
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"No existing temp file for batch {batch_index}, column '{column}'. Starting fresh.",
            level="info",
        )

    gemma = GemmaHandler()

    # 2) Determine which names still need processing
    to_process = [name for name in batch_values if name not in batch_mapping]
    if not to_process:
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"No names left to process in batch {batch_index} for column '{column}'.",
            level="info",
        )
        return  # nothing left in this batch

    # 3) Build JSON payload for all to_process names
    payload_dict = {"standardized_data": [{"raw_input": name} for name in to_process]}
    custom_input = json.dumps(payload_dict)

    attempts = 0
    response = None
    while attempts < 3:
        attempts += 1
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Sending request to Gemma for batch {batch_index}, column '{column}', "
                f"attempt {attempts}/3 with {len(to_process)} names."
            ),
            level="info",
        )
        try:
            response = gemma.process_prompt(STANDARDIZER_PROMPT, custom_input)
        except Exception as e:
            log_message(
                folder=folder,
                raw_manifest_filename=raw_manifest_filename,
                log_string=(
                    f"Gemma exception for batch {batch_index}, column '{column}', "
                    f"attempt {attempts}/3: {e}. Retrying in 5s."
                ),
                level="error",
            )
            time.sleep(5)
            continue

        # Now response should be a dict with top-level "standardized_data"
        
        if (
            isinstance(response, dict)
            and "standardized_data" in response
            and isinstance(response["standardized_data"], list)
            and len(response["standardized_data"]) == len(to_process)
        ):
            valid = True
            for item in response["standardized_data"]:
                if not (
                    isinstance(item, dict)
                    and "raw_input" in item
                    and "output" in item
                ):
                    valid = False
                    break
            if valid:
                log_message(
                    folder=folder,
                    raw_manifest_filename=raw_manifest_filename,
                    log_string=(
                        f"Received valid response from Gemma for batch {batch_index}, column '{column}'."
                    ),
                    level="info",
                )
                break

        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Invalid response structure from Gemma for batch {batch_index}, "
                f"column '{column}', attempt {attempts}/3. Retrying in 2s."
            ),
            level="error",
        )
        time.sleep(2)

    # 4) Build final mapping (either from response or fallback)
    final_mapping = {}
    if response and isinstance(response, dict) and "standardized_data" in response:
        for item in response["standardized_data"]:
            raw = item.get("raw_input", "")
            cleaned = item.get("output", raw)
            final_mapping[raw] = cleaned
    else:
        if city_flag == False:
            for raw in to_process:
                final_mapping[raw] = raw
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Max retries reached or invalid JSON for batch {batch_index}, column '{column}'. "
                f"Falling back to pre-cleaned names."
            ),
            level="error",
        )

    batch_mapping.update(final_mapping)

    # 5) Persist to temp file
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(batch_mapping, f, indent=2)
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Written temp file for batch {batch_index}, column '{column}', "
                f"total entries: {len(batch_mapping)}."
            ),
            level="info",
        )
    except Exception as e:
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Error writing temp file for batch {batch_index}, column '{column}': {e}"
            ),
            level="error",
        )


def standardize_data(
    dataframe: pd.DataFrame, raw_manifest_filename: str, STANDARDIZER_PROMPT: str, COLUMNS_TO_STANDARDIZE: list[str], FOLDER_NAME: str, city_flag = False
) -> pd.DataFrame:
    """
    Standardize each column in COLUMNS_TO_STANDARDIZE by:
    1) Local cleanup → pre_cleaned_<column>.
    2) Deduplicate and split unique values into batches of DATA_CHUNK_SIZE.
    3) Compute num_threads = min(GEMMA_NUM_THREADS, total_batches).
    4) Use ThreadPoolExecutor with dynamic num_threads to process batches in parallel,
       wrapped in a tqdm progress bar.
    5) Merge all batch mappings, build cleaned_<column>, drop/rename.
    """
    folder = FOLDER_NAME
    log_message(
        folder=folder,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Starting standardizing {folder}.",
        level="info",
    )

    if "ID" not in dataframe.columns:
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string="DataFrame missing 'ID' column. Cannot proceed.",
            level="error",
        )
        raise ValueError("DataFrame must contain an 'ID' column for mapping.")

    base_temp_folder = os.path.join(TEMP_DIR, folder)
    os.makedirs(base_temp_folder, exist_ok=True)

    for column in COLUMNS_TO_STANDARDIZE:
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=f"Standardizing column '{column}'.",
            level="info",
        )

        # 1) Local cleanup
        pre_col = f"pre_cleaned_{column}"
        dataframe[pre_col] = dataframe[column].apply(local_clean_name)
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Completed local cleanup for column '{column}'. "
                f"Generated '{pre_col}' with {dataframe[pre_col].notna().sum()} entries."
            ),
            level="info",
        )

        # 2) Build unique list (exclude empty strings)
        unique_values = dataframe[pre_col].dropna().astype(str).unique().tolist()
        unique_values = [u for u in unique_values if u != ""]
        num_unique = len(unique_values)

        if num_unique == 0:
            log_message(
                folder=folder,
                raw_manifest_filename=raw_manifest_filename,
                log_string=f"No non-empty values to standardize for column '{column}'. Dropping column.",
                level="info",
            )
            dataframe.drop(columns=[column], inplace=True)
            continue

        # 3) Create batches of size DATA_CHUNK_SIZE
        batch_lists = [
            unique_values[i : i + DATA_CHUNK_SIZE]
            for i in range(0, num_unique, DATA_CHUNK_SIZE)
        ]
        total_batches = len(batch_lists)
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Created {total_batches} batches (chunk size {DATA_CHUNK_SIZE}) "
                f"for column '{column}' with {num_unique} unique values."
            ),
            level="info",
        )

        # 4) Determine how many threads to launch
        num_threads = min(GEMMA_NUM_THREADS, total_batches)
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Configuring ThreadPoolExecutor with {num_threads} threads for column '{column}'."
            ),
            level="info",
        )

        # 5) Use ThreadPoolExecutor with tqdm to track progress
        final_mapping: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {}
            for i, batch in enumerate(batch_lists):
                future = executor.submit(
                    process_batch, batch, column, i, raw_manifest_filename, STANDARDIZER_PROMPT, folder, city_flag
                )
                future_to_index[future] = i

            for future in tqdm(
                as_completed(future_to_index),
                total=total_batches,
                desc=f"Processing {column}",
                ncols=80,
            ):
                idx = future_to_index[future]
                try:
                    future.result()
                    log_message(
                        folder=folder,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Batch {idx} for column '{column}' completed successfully.",
                        level="info",
                    )
                except Exception as e:
                    log_message(
                        folder=folder,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=(
                            f"Unexpected error in batch {idx}, column '{column}': {e}"
                        ),
                        level="error",
                    )

        # 6) Merge all batch JSONs into a single mapping
        for i in range(total_batches):
            temp_file = os.path.join(
                base_temp_folder, f"temp_{column}_cleaned_batch_{i}.json"
            )
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    final_mapping.update(data)
                except Exception as e:
                    log_message(
                        folder=folder,
                        raw_manifest_filename=raw_manifest_filename,
                        log_string=f"Error reading {temp_file}: {e}",
                        level="error",
                    )
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Merged mappings for column '{column}'. Total mappings: {len(final_mapping)}."
            ),
            level="info",
        )

        # 7) Build cleaned column by mapping pre_cleaned → final; fallback to pre_cleaned
        
        if city_flag:
            cleaned_col = f"{column}_city"
        else:
            cleaned_col = f"cleaned_{column}"

        dataframe[cleaned_col] = dataframe[pre_col].map(final_mapping)
        dataframe[cleaned_col] = dataframe[cleaned_col].fillna(dataframe[pre_col])
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Built '{cleaned_col}' for column '{column}'. "
                f"Entries filled: {dataframe[cleaned_col].notna().sum()}."
            ),
            level="info",
        )

        # 8) Drop old columns & rename
        dataframe.drop(columns=[column, pre_col], inplace=True)
        if city_flag:
            # Remove "Address" and create suffix column
            base_name = column.replace("Address", "").strip().lower()
            new_col = f"{base_name} City"
            dataframe.rename(columns={cleaned_col: new_col}, inplace=True)
        else:
            dataframe.rename(columns={cleaned_col: column}, inplace=True)
        log_message(
            folder=folder,
            raw_manifest_filename=raw_manifest_filename,
            log_string=(
                f"Dropped original column '{column}' and '{pre_col}', renamed '{cleaned_col}' to '{column}'."
            ),
            level="info",
        )
    
    log_message(
        folder=folder,
        raw_manifest_filename=raw_manifest_filename,
        log_string=f"Completed standardizing {folder}.",
        level="info",
    )
    return dataframe