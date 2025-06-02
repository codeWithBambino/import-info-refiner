import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from src.config.config import CREDENTIALS_JSON

def read_google_sheet(
    sheet_id_or_url: str,
    worksheet_name: str
) -> pd.DataFrame:
    """
    Reads the specified worksheet from a Google Sheet and returns a pandas DataFrame.
    
    Args:
      sheet_id_or_url: The spreadsheet ID (e.g. "1AbCd…") or full URL.
      worksheet_name:  The exact name/tab of the worksheet.
      cred_json_path:  Path to the service account JSON key file.
    
    Returns:
      DataFrame containing all rows (headers → columns).
    """
    # 1. Define OAuth scope and authorize
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_JSON, scope)
    client = gspread.authorize(creds)

    # 2. Open sheet by URL or by key
    if sheet_id_or_url.startswith("http"):
        spreadsheet = client.open_by_url(sheet_id_or_url)
    else:
        spreadsheet = client.open_by_key(sheet_id_or_url)

    # 3. Select worksheet and fetch all records
    worksheet = spreadsheet.worksheet(worksheet_name)
    records = worksheet.get_all_records()

    # 4. Convert to DataFrame and return
    dataframe = pd.DataFrame(records)
    return dataframe