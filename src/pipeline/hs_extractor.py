import re
import os
import pandas as pd
import json
from src.config.config import REFERENCE_DIR
from src.helpers.logger import log_message
from src.config.folder_name import HS_CODE_EXTRACTION_FOLDER

HS_FILE_PATH = os.path.join(REFERENCE_DIR, "hscodes.json")
# -- Regex patterns as above --
HS_CODE_PATTERNS = [
    r'(?:(?:HSN?|HTS|H\.?S\.?|HARMONIZED(?:\s+SYSTEM)?|CUSTOMS|TARIFF)\s*(?:CODE|NO\.?|NUMBER)?[\s:=#\-]*)(\d{4}\.\d{2}(?:\.\d{2,4})?|\d{6,10})\b',
    r'\b(\d{4}\.\d{2}(?:\.\d{2,4})?|\d{6,10})\s+(?:IS\s+)?(?:THE\s+)?(?:HSN?|HTS|H\.?S\.?)\s*(?:CODE|NO\.?|NUMBER)?\b',
    r'\b(\d{4}\.\d{2}\.\d{2}(?:\.\d{2})?)\b',
    r'\b(\d{4}\.\d{2})\b',
    r'\b(\d{10})\b',
    r'\b(\d{8})\b',
    r'\b(\d{6})\b',
    r'(?:HS(?:N)?(?:\s*CODE)?(?:\s*NO)?[\s.:=]*)(\d{6,10})',
    r'\b(\d{6,10})[\s/-]+\d+[\s/-]*\d*\b',
    r'\b(\d{4,10})\b',
]

def extract_hs_codes(text):
    if pd.isna(text):
        return []
    found = []
    for pattern in HS_CODE_PATTERNS:
        matches = re.findall(pattern, str(text), re.IGNORECASE)
        for match in matches:
            clean = match.strip().replace(" ", "")
            if clean not in found:
                found.append(clean)
    return found

def hs_code_verifier(found, hs_json_path=HS_FILE_PATH):
    with open(hs_json_path, 'r') as f:
        hs_list = set(json.load(f)["hscode"])
    return [code for code in found if code in hs_list]

def extract_hs_code(dataframe: pd.DataFrame, raw_manifest_filename: str, hs_json_path=HS_FILE_PATH) -> pd.DataFrame:
    log_message(HS_CODE_EXTRACTION_FOLDER, raw_manifest_filename, "Starting HS Code extraction process.", level="info")
    try:
        dataframe["HS Code List"] = dataframe["Commodity"].apply(extract_hs_codes)
        # Verify using the loaded JSON list
        dataframe["Verified HS Codes"] = dataframe["HS Code List"].apply(lambda x: hs_code_verifier(x, hs_json_path) if hs_code_verifier(x, hs_json_path) else "")
        dataframe.drop(columns=["HS Code List"], inplace=True)
        log_message(HS_CODE_EXTRACTION_FOLDER, raw_manifest_filename, "HS Code extraction process completed successfully.", level="info")
    except Exception as e:
        log_message(HS_CODE_EXTRACTION_FOLDER, raw_manifest_filename, f"Error during HS Code extraction: {str(e)}", level="error")
        raise
    return dataframe