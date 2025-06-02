import os
import requests
import json
import time # Added for retry delay
from typing import Dict, Any, Optional
from src.config.config import GEMMA_HOST, MODEL
from src.helpers.logger import setup_logger

class GemmaHandler:
    def __init__(self, log_folder: str = 'gemma'):
        """Initialize GemmaHandler with logging setup.

        Args:
            log_folder (str): Subfolder name for logs (default: 'gemma')
        """
        self.logger = setup_logger(log_folder, 'gemma_api.log')

    def _prompt_retriever(self, template_path: str, custom_input: str) -> str:
        """Read a template file and replace placeholder with custom input.

        Args:
            template_path (str): Path to the template file containing the placeholder
            custom_input (str): The content to replace the placeholder with

        Returns:
            str: The updated prompt with the custom input inserted

        Raises:
            FileNotFoundError: If the template file does not exist
            IOError: If there are issues reading the template file
            ValueError: If the template is invalid or missing required placeholder
        """
        try:
            # Validate template path
            if not os.path.exists(template_path):
                error_msg = f"Template file not found: {template_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Read the template file
            with open(template_path, 'r', encoding='utf-8') as file:
                template_content = file.read()

            # Validate template contains required placeholder
            if "{gemma_custom_input}" not in template_content:
                error_msg = "Template is missing required {gemma_custom_input} placeholder"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Replace the placeholder with the custom input
            updated_prompt = template_content.replace("{gemma_custom_input}", str(custom_input))
            self.logger.info(f"Successfully generated prompt from template: {template_path}")

            return updated_prompt

        except Exception as e:
            self.logger.error(f"Error in prompt retriever: {str(e)}", exc_info=True)
            raise

    def _ask_gemma(self, prompt: str, retries: int = 3, delay: int = 5) -> Dict[str, Any]:
        """Send a prompt to the Gemma model using OpenAI-compatible API.

        Args:
            prompt (str): The prompt/question to ask.
            retries (int): Number of times to retry the request in case of failure.
            delay (int): Delay in seconds between retries.

        Returns:
            Dict[str, Any]: Response dictionary containing status_code and other response data
        """
        # Create payload in OpenAI format
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer ollama"  # Using "ollama" as the API key
        }

        # Construct the full URL for chat completions
        endpoint = f"{GEMMA_HOST}/chat/completions"

        last_exception = None

        for attempt in range(retries):
            try:
                self.logger.debug(f"Sending prompt to Gemma model {MODEL} (Attempt {attempt + 1}/{retries}): {prompt[:50]}...")
                response = requests.post(endpoint, json=payload, headers=headers, timeout=60) # Added timeout

                # Check if the request was successful
                if response.status_code == 200:
                    response_data = response.json()

                    # Extract content from the response
                    if response_data.get("choices") and len(response_data["choices"]) > 0:
                        content = response_data["choices"][0].get("message", {}).get("content", "")

                        try:
                            # Find JSON content (sometimes models wrap JSON in markdown code blocks)
                            if "```json" in content:
                                json_text = content.split("```json")[1].split("```")[0].strip()
                            elif "```" in content:
                                json_text = content.split("```")[1].split("```")[0].strip()
                            else:
                                json_text = content.strip()

                            result = json.loads(json_text)
                            self.logger.debug("Successfully parsed JSON response")
                            return {"status": "success", "status_code": 200, "data": result}
                        except json.JSONDecodeError as je:
                            self.logger.warning(f"Failed to parse JSON response: {je}")
                            # Don't retry on JSON parse error, as it's a model response issue
                            return {
                                "status": "error",
                                "status_code": 400,
                                "error": "JSON_PARSE_ERROR",
                                "message": "Model did not return valid JSON",
                                "raw_response": content
                            }
                    else:
                        error_msg = "No choices found in response"
                        self.logger.error(error_msg)
                        # Don't retry if no choices, likely a persistent issue or bad prompt
                        return {
                            "status": "error",
                            "status_code": 500,
                            "error": "RESPONSE_ERROR",
                            "message": error_msg,
                            "raw_response": response.text
                        }
                else:
                    error_msg = f"Request failed with status code {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    # Retry for server-side errors (5xx) or specific client errors if appropriate
                    if 500 <= response.status_code < 600:
                        last_exception = requests.exceptions.HTTPError(error_msg)
                        self.logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue # Retry the loop
                    else:
                        # For other client errors (4xx), don't retry unless specific ones are known to be transient
                        return {
                            "status": "error",
                            "status_code": response.status_code,
                            "error": "API_CLIENT_ERROR",
                            "message": error_msg
                        }

            except requests.exceptions.RequestException as e:
                error_msg = f"Error talking to Gemma (Attempt {attempt + 1}/{retries}): {str(e)}"
                self.logger.error(error_msg, exc_info=False) # exc_info=False to avoid repetitive tracebacks in logs for retries
                last_exception = e
                if attempt < retries - 1:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {retries} retries failed.")
                    break # Exit loop after last attempt
        
        # If all retries failed, return the last known error
        final_error_msg = f"Error talking to Gemma after {retries} retries: {str(last_exception)}"
        self.logger.error(final_error_msg, exc_info=True if last_exception else False)
        return {
            "status": "error",
            "status_code": 500, # Generic server error after retries
            "error": "API_CONNECTION_ERROR",
            "message": final_error_msg
        }

    def process_prompt(self, template_path: str, custom_input: str) -> Dict[str, Any]:
        """Process a prompt using a template and custom input.

        Args:
            template_path (str): Path to the prompt template file
            custom_input (str): Custom input to insert into the template

        Returns:
            Dict[str, Any]: Response dictionary containing either the success response data
            or error information with status and message.
        """
        try:
            # Get the prompt from template
            try:
                prompt = self._prompt_retriever(template_path, custom_input)
            except Exception as e:
                error_msg = f"Failed to process prompt template: {str(e)}"
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "error": "TEMPLATE_PROCESSING_ERROR",
                    "message": error_msg
                }

            # Call Gemma API
            response = self._ask_gemma(prompt)

            # Log the response based on status
            if isinstance(response, dict) and response.get('status') == 'success':
                self.logger.info("Successfully processed prompt with Gemma API")
                return response
            else:
                error_msg = response.get('message', 'Unknown error occurred')
                self.logger.error(f"Gemma API request failed: {error_msg}")
                return {
                    "status": "error",
                    "error": "API_ERROR",
                    "message": error_msg
                }

        except Exception as e:
            error_msg = f"Unexpected error in process_prompt: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "error": "PROCESSING_ERROR",
                "message": error_msg
            }

    def extract_data(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and validate data from a successful Gemma response.

        Args:
            response_data (Dict[str, Any]): The response data from process_prompt

        Returns:
            Optional[Dict[str, Any]]: Extracted data if successful, None if invalid
        """
        try:
            if response_data.get('status') == 'success' and 'data' in response_data:
                return response_data['data']
            return None
        except Exception as e:
            self.logger.error(f"Error extracting data from response: {str(e)}")
            return None


