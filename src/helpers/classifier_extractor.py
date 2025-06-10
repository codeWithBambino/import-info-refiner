import pandas as pd

def add_category_column(dataframe: pd.DataFrame,
                        target_classification_column: str,
                        classified_json: list[dict]) -> pd.DataFrame:
    """
    Adds a new column "<target_classification_column> Category" to dataframe,
    using classified_json to look up the category for each company name.

    - dataframe: your DataFrame
    - target_classification_column: name of the column containing company names
    - classified_json: list of {"Company Name": ..., "Category": ...}
    """
    # 1. Build mapping dict
    mapping = {
        entry["company_name"]: entry["category"]
        for entry in classified_json["categorized_data"]
    }

    # 2. Create & populate the new category column
    new_col = f"{target_classification_column} Category"
    dataframe[new_col] = (
        dataframe[target_classification_column]
        .map(mapping)              # look up each value
        .fillna("")         # or leave as NaN if you prefer
    )

    return dataframe

