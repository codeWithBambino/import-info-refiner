from google import genai
from google.genai import types 
from src.config.config import GEMINI_MODEL, GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)
search_tool = types.Tool(google_search={})

def ask_gemini(prompt):
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[search_tool],
            ),
        )

        return response.text

    except genai.errors.APIError as e: # Handles API errors [7]
        print(f"An API error occurred: {e.code} - {e.message}") # [7]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
