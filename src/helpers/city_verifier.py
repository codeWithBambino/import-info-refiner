import pandas as pd
import json
from src.config.config import CITIES_JSON_PATH

def city_verifier(df, columns):
   try:
       # Load and validate JSON file
       try:
           with open(CITIES_JSON_PATH, 'r') as file:
               city_json = json.load(file)
       except FileNotFoundError:
           print(f"Error: Cities JSON file not found at {CITIES_JSON_PATH}")
           return df
       except json.JSONDecodeError:
           print(f"Error: Invalid JSON format in {CITIES_JSON_PATH}")
           return df
       except Exception as e:
           print(f"Error loading JSON file: {str(e)}")
           return df

       # Validate input parameters
       if df is None or df.empty:
           print("Error: DataFrame is None or empty")
           return df
       
       # Convert columns to list if it's a single string
       if isinstance(columns, str):
           columns = [columns]
       elif not isinstance(columns, list):
           print("Error: columns parameter must be a string or list of strings")
           return df
       
       # Validate all columns exist in DataFrame
       missing_columns = [col for col in columns if col not in df.columns]
       if missing_columns:
           print(f"Error: Column(s) {missing_columns} not found in DataFrame")
           return df

       # Preprocess JSON keys for fast lookup
       try:
           normalized_json = {}
           for city, value in city_json.items():
               if not isinstance(value, dict) or "State" not in value or "PIN" not in value:
                   print(f"Warning: Invalid data structure for city '{city}', skipping")
                   continue
               normalized_key = str(city).strip().upper().replace(" ", "")
               normalized_json[normalized_key] = value
       except Exception as e:
           print(f"Error preprocessing JSON data: {str(e)}")
           return df
       
       # Process each column
       for column in columns:
           try:
               print(f"Processing column: {column}")
               
               # Prepare output columns for current column
               state_col = f"{column} State"
               pin_col = f"{column} PIN"
               
               try:
                   df[state_col] = ""
                   df[pin_col] = ""
               except Exception as e:
                   print(f"Error creating output columns for {column}: {str(e)}")
                   continue

               # Process each record in current column
               try:
                   for idx, val in df[column].items():
                       try:
                           # Handle null/empty values
                           if pd.isnull(val) or val == "":
                               df.at[idx, column] = ""
                               continue
                           
                           # Normalize the city from dataframe
                           norm_val = str(val).strip().upper().replace(" ", "")
                           match = normalized_json.get(norm_val)
                           
                           if match:
                               try:
                                   df.at[idx, state_col] = str(match["State"])
                                   # Handle PIN data safely
                                   pins = match["PIN"]
                                   if isinstance(pins, list):
                                       df.at[idx, pin_col] = ",".join(map(str, pins))
                                   else:
                                       df.at[idx, pin_col] = str(pins)
                               except KeyError as e:
                                   print(f"Warning: Missing key {e} for city '{val}' at index {idx} in column {column}")
                                   df.at[idx, column] = ""
                               except Exception as e:
                                   print(f"Warning: Error processing match for '{val}' at index {idx} in column {column}: {str(e)}")
                                   df.at[idx, column] = ""
                           else:
                               df.at[idx, column] = ""  # Clear cell if not matched
                               
                       except Exception as e:
                           print(f"Warning: Error processing row {idx} with value '{val}' in column {column}: {str(e)}")
                           continue
                           
               except Exception as e:
                   print(f"Error during data processing for column {column}: {str(e)}")
                   continue
                   
               print(f"Completed processing column: {column}")
               
           except Exception as e:
               print(f"Error processing column {column}: {str(e)}")
               continue

       return df
       
   except Exception as e:
       print(f"Unexpected error in city_verifier: {str(e)}")
       return df