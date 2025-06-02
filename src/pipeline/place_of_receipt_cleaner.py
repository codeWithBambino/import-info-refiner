import re
import os
import pandas as pd
from src.helpers.logger import log_message
from src.config.folder_name import STANDARDIZE_PLACE_FOLDER

def clean_place_name_regex(raw):
    if not isinstance(raw, str):
        return raw

    original = raw
    raw = raw.upper().strip()

    # Remove noise words
    raw = re.sub(r'\b(INDIA|IN|PB|HR|UP|MP|ICD|CFS|PORT|SEA|TERMINAL|CONCOR|GJ|HR|DL)\b', '', raw)
    raw = re.sub(r'[.,\-()/]', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    # Replace known bad abbreviations or incomplete codes
    if raw in {"M", "HLCU", "RAIL", "HIND", "TUGHL", "JAWAHARLAL", "GRFL"}:
        return "UNKNOWN"
    if raw == "INMUN":
        return "MUNDRA"

    # Pattern mapping
    pattern_map = [
        (r'NAHVA SHEVA.*', 'NHAVA SHEVA'),
        (r'NHAVA SHEVA.*', 'NHAVA SHEVA'),
        (r'JAWAHARLAL.*', 'NHAVA SHEVA'),
        (r'MUMBAI.*', 'MUMBAI'),
        (r'LUDHIANA.*', 'LUDHIANA'),
        (r'SAHNEWAL.*', 'SAHNEWAL'),
        (r'DADRI.*', 'DADRI'),
        (r'TUGHLAKABAD.*', 'TUGHLAKABAD'),
        (r'MORADABAD.*', 'MORADABAD'),
        (r'MANDIDEEP.*', 'MANDIDEEP'),
        (r'KILA RAIPUR.*', 'KILA RAIPUR'),
        (r'CHAWAPAYAL.*', 'CHAWAPAYAL'),
        (r'CHAWAPAIL.*', 'CHAWAPAYAL'),
        (r'JODHPUR.*', 'JODHPUR'),
        (r'PIPAVAV.*', 'PIPAVAV'),
        (r'MUNDRA.*', 'MUNDRA'),
        (r'TUTICORIN.*', 'TUTICORIN'),
        (r'KOLKATA.*', 'KOLKATA'),
        (r'KOLKATA CALCUTTA', 'KOLKATA'),
        (r'SHANGHAI.*', 'SHANGHAI'),
        (r'QINGDAO.*', 'QINGDAO'),
        (r'NINGBO.*', 'NINGBO'),
        (r'YANTIAN.*', 'YANTIAN'),
        (r'BUSAN.*', 'BUSAN'),
        (r'SINGAPORE.*', 'SINGAPORE'),
        (r'FREEPORT.*', 'FREEPORT'),
        (r'HALDIA.*', 'HALDIA'),
        (r'HAMBURG.*', 'HAMBURG'),
        (r'VALENCIA.*', 'VALENCIA'),
        (r'BARCELONA.*', 'BARCELONA'),
        (r'ROTTERDAM.*', 'ROTTERDAM'),
        (r'SALALAH.*', 'SALALAH'),
        (r'LE HAVRE.*', 'LE HAVRE'),
        (r'CAUCEDO.*', 'CAUCEDO'),
        (r'EDMONTON.*', 'EDMONTON'),
        (r'CALGARY.*', 'CALGARY'),
        (r'TORONTO.*', 'TORONTO'),
        (r'VANCOUVER.*', 'VANCOUVER'),
        (r'BOSTON.*', 'BOSTON'),
        (r'MONTREAL.*', 'MONTREAL'),
        (r'MIAMI.*', 'MIAMI'),
        (r'MOBILE.*', 'MOBILE'),
        (r'QUERETARO.*', 'QUERETARO'),
        (r'MEXICO CITY.*', 'MEXICO CITY'),
        (r'APODACA.*', 'APODACA'),
        (r'MONTERREY.*', 'MONTERREY'),
        (r'LONDON.*', 'LONDON'),
        (r'HITCHIN.*', 'HITCHIN'),
        (r'CROYDON.*', 'CROYDON'),
        (r'CAMBRIDGE.*', 'CAMBRIDGE'),
        (r'GRAVELEY.*', 'GRAVELEY'),
        (r'ROYSTON.*', 'ROYSTON'),
        (r'BECCLES.*', 'BECCLES'),
        (r'HIND TERMINAL.*', 'HIND TERMINAL ICD'),
        (r'GATEWAY.*', 'GATEWAY TERMINAL'),
        (r'GRFL.*', 'LUDHIANA GRFL'),
        (r'KLPPL.*', 'PANKI'),
        (r'KANECH.*', 'KANECH'),
        (r'KHODIYAR.*', 'KHODIYAR'),
        (r'SAMALKHA.*', 'SAMALKHA'),
        (r'JATTIPUR.*', 'JATTIPUR'),
        (r'NEW DELHI.*', 'NEW DELHI'),
        (r'DELHI.*', 'NEW DELHI'),
    ]

    for pattern, replacement in pattern_map:
        if re.match(pattern, raw):
            return replacement

    return raw

def standardize_place_of_receipt(dataframe: pd.DataFrame, column_name:str ,raw_manifest_filename: str) -> pd.DataFrame:
    if 'Place of Receipt' not in dataframe.columns:
        log_message(STANDARDIZE_PLACE_FOLDER, raw_manifest_filename, 'Missing column: Place of Receipt', level="error")
        return dataframe

    matched, unmatched = [], []

    for idx, val in dataframe[column_name].items():
        cleaned = clean_place_name_regex(val)
        if cleaned != str(val).strip().upper():
            matched.append({"RowIndex": idx, "Original": val, "Cleaned": cleaned})
        else:
            unmatched.append({"RowIndex": idx, "Original": val, "Cleaned": cleaned})

        dataframe.at[idx, 'Place of Receipt'] = cleaned

    os.makedirs(os.path.join('logs', STANDARDIZE_PLACE_FOLDER), exist_ok=True)

    if unmatched:
        pd.DataFrame(unmatched).drop_duplicates(subset=['Original', 'Cleaned']).to_csv(f'logs/{STANDARDIZE_PLACE_FOLDER}/unmatched_{raw_manifest_filename}', index=False)

    return dataframe