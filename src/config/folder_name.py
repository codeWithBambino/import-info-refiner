# Log folder names used across the application

PIPELINE_MAIN_PROCESS_FOLDER = 'pipeline_main_process'

# Step 1: Remove exact duplicates
REMOVE_DUPLICATES_FOLDER = '1_remove_duplicates'

# Step 2: Deduplicate by Master BOL and Container
DEDUPLICATE_MBL_CONTAINER_FOLDER = '2_deduplicate_mbl_container'

# Step 3: Map SCAC to LSP
MAP_SCAC_FOLDER = '3_map_scac_to_lsp'

# Step 4: Standardize Party Names
STANDARDIZE_PARTY_NAMES_FOLDER = '4_standardize_party_names'

# Step 5: Standardize Place of Receipt
STANDARDIZE_PLACE_FOLDER = "5_standardize_place_of_receipt"

CITY_EXTRACTION_FOLDER = "6_7_city_extraction"

# Step 6: Extract Cities from Shipper Address
CITY_EXTRACTION_FOLDER = "6_shipper_city_extraction"

# Step 7: Standardize Cities from Consignee Address
CONSIGNEE_CITY_EXTRACTION_FOLDER = "7_consignee_city_extraction"

# Step 8: Shipper Entity Classification
SHIPPER_ENTITY_CLASSIFICATION_FOLDER = "8_shipper_entity_classification"
CONSIGNEE_CITY_EXTRACTION_FOLDER = "8_consignee_entity_classification"
LSP_ENTITY_CLASSIFICATION_FOLDER = "8_lsp_entity_classification"

# Step 11: HS Code Extraction
HS_CODE_EXTRACTION_FOLDER = "11_hs_code_extraction"