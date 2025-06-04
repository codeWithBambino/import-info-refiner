import os

# Root project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Gemma
GEMMA_HOST = "http://3.109.55.66:11434/v1"
MODEL = "gemma3:12b"

# Chunking configs
GEMMA_NUM_THREADS = 6
DATA_CHUNK_SIZE = 10

# Gemma Prompt Paths
PARTY_STANDARDIZER_PROMPT = os.path.join(BASE_DIR, "src/prompts/party_standardizer_prompt.txt")
CITY_EXTRACTION_PROMPT = os.path.join(BASE_DIR, "src/prompts/city_extraction_prompt.txt")

# Input paths
INPUT_MANIFESTS = os.path.join(BASE_DIR, "data/manifests/raw")
PROCESS_MANIFESTS = os.path.join(BASE_DIR, "data/manifests/processing")
REFERENCE_DIR = os.path.join(BASE_DIR, "data/reference")

# Columns to standardize
COLUMNS_TO_STANDARDIZE = ['Shipper', 'Consignee', 'Notify Party 1', 'Notify Party 2']

# Output paths
OUTPUT_CLEANED = os.path.join(BASE_DIR, "data/manifests/cleaned")

# Temp path
TEMP_DIR = os.path.join(BASE_DIR, "data/temp")

# Log path (if needed for future reference)
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Cities json path
CITIES_FILE = os.path.join(BASE_DIR, "data/reference/cities.json")

# Entity Category json path
ENTITY_CATEGORY_FILE = os.path.join(BASE_DIR, "data/reference/entity_category.json")

# HS Code json path
HS_CODE_FILE = os.path.join(BASE_DIR, "data/reference/hscodes.json")

# Ports json path
PORTS_FILE = os.path.join(BASE_DIR, "data/reference/ports.json")

# Manual Validation path
MANUAL_VALIDATION_DIR = os.path.join(BASE_DIR, "manual_validation")

# Test data path
RAW_TEST_DATA = os.path.join(BASE_DIR, "data/test_data/raw")
CLEANED_TEST_DATA_FOLDER = os.path.join(BASE_DIR, "data/test_data/cleaned")
VERIFIED_TEST_DATA = os.path.join(BASE_DIR, "data/test_data/verified/verified_test_dataset.csv")
ACCURACY_ANALYTICS = os.path.join(BASE_DIR, "tests/accuracy_analytics.csv")

# Google
CREDENTIALS_JSON = os.path.join(BASE_DIR, "src/config/google.json")
IMPORT_INFO_TEST_DATASET_URL = "https://docs.google.com/spreadsheets/d/1sCzrkpxJoj0KxIsywDKiZLMfVCuLujuyMtyIZU1hgQc/edit?usp=sharing"
VERIFIED_SHEET_NAME = "VERIFIED"
RAW_SHEET_NAME = "RAW"