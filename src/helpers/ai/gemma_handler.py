# Gemma
import re
import requests
import json
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config.config import GEMMA_HOST, MODEL, GEMMA_NUM_THREADS
from src.helpers.logger import setup_logger


class GemmaHandler:
    def __init__(self, log_folder: str = 'gemma'):
        """
        Initialize GemmaHandler with logging setup.

        Args:
            log_folder (str): Subfolder name for logs (default: 'gemma')
        """
        self.logger = setup_logger(log_folder, 'gemma_api.log')

    def _prompt_retriever(self, template_path: str, custom_input: str) -> str:
        """
        Read a template file and replace placeholders with custom_input.

        Supports both {{INPUT}} and {gemma_custom_input} placeholders.

        Args:
            template_path (str): Path to the text file containing placeholders.
            custom_input (str): The string to inject into those placeholders.

        Returns:
            str: A fully formed prompt ready to send to Gemma.
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            # Replace both placeholder syntaxes
            updated = template_content.replace("{{INPUT}}", custom_input)
            updated = updated.replace("{gemma_custom_input}", custom_input)
            return updated
        except Exception as e:
            self.logger.error(f"_prompt_retriever: could not read '{template_path}': {e}")
            raise

    def _ask_gemma(self, prompt: str, retries: int = 3, delay: int = 2) -> Dict[str, Any]:
        """
        Send the prompt to Gemma’s /chat/completions endpoint and return its JSON response.

        This method logs the raw request payload and raw response JSON with a blank line between
        for easier debugging.

        Args:
            prompt (str): The prompt string to send.
            retries (int): How many times to retry on non-2xx or network error.
            delay (int): Base backoff in seconds (doubles each retry).

        Returns:
            Dict[str, Any]: If successful, the JSON-decoded Gemma response;
                            otherwise a dict with 'status':'error' and a message.
        """
        url = f"{GEMMA_HOST}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        for attempt in range(1, retries + 1):
            try:
                # Log the raw request payload
                self.logger.info(f"REQUEST: {json.dumps(payload, ensure_ascii=False)}\n")
                self.logger.info(f"→ [Attempt {attempt}/{retries}] POST to {url}")
                start_time = time.time()

                response = requests.post(url, headers=headers, json=payload, timeout=60)

                elapsed = time.time() - start_time
                status = response.status_code

                # Log the raw response JSON
                try:
                    text = response.text
                    self.logger.info(f"\nRESPONSE: {text}\n")
                except Exception as e:
                    self.logger.error(f"Failed to read response text: {e}")

                self.logger.info(f"← [Attempt {attempt}/{retries}] HTTP {status} (took {elapsed:.1f}s)")

                if 200 <= status < 300:
                    try:
                        return response.json()
                    except json.JSONDecodeError as je:
                        self.logger.error(f"JSON decode error from Gemma: {je}")
                        return {"status": "error", "message": "Invalid JSON from Gemma"}
                else:
                    text_snippet = response.text[:200]
                    self.logger.warning(
                        f"Attempt {attempt} returned HTTP {status}: {text_snippet}"
                    )

            except requests.exceptions.RequestException as rexc:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"⚠ [Attempt {attempt}/{retries}] network error after {elapsed:.1f}s: {rexc}"
                )

            if attempt < retries:
                sleep_time = delay * (2 ** (attempt - 1))
                self.logger.info(f"⏳ Sleeping {sleep_time}s before retry #{attempt+1}")
                time.sleep(sleep_time)

        err_msg = "Gemma API request failed after retries"
        self.logger.error(err_msg)
        return {"status": "error", "message": err_msg}

    def extract_data(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse the chat-completions response, extract the assistant's content,
        strip markdown/code block if present, and convert it to a Python dict.
        """
        try:
            choices = response_data.get("choices")
            if not choices or not isinstance(choices, list):
                self.logger.error("extract_data: 'choices' missing or not a list")
                return None

            assistant_msg = choices[0].get("message", {}).get("content", "")
            if not isinstance(assistant_msg, str) or not assistant_msg.strip():
                self.logger.error("extract_data: assistant 'content' is empty or not a string")
                return None

            # 🟡 NEW: If wrapped in code block, extract only the code part
            # Handles ```json ... ``` or just ``` ... ```
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", assistant_msg, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = assistant_msg

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as je:
                self.logger.error(f"extract_data: Could not parse assistant content as JSON: {je}")
                return None

            if "Companies" in parsed and isinstance(parsed["Companies"], list):
                return parsed

            self.logger.error("extract_data: parsed JSON has no 'Companies' key or it's not a list")
            return None

        except Exception as e:
            self.logger.error(f"extract_data: Unexpected exception: {e}")
        return None

    def process_prompt(self, template_path: str, custom_input: str) -> Optional[Dict[str, Any]]:
        """
        High-level wrapper to build a prompt (via the template), send to Gemma,
        and return the parsed JSON from the assistant (with "Companies").

        Args:
            template_path (str): Path to the .txt template file with placeholders.
            custom_input (str): The chunk of names or instructions to inject.

        Returns:
            Optional[Dict[str,Any]]: The parsed JSON dict returned by Gemma’s assistant,
                                     or None if any step fails.
        """
        try:
            final_prompt = self._prompt_retriever(template_path, custom_input)
            self.logger.info("Sending prompt to Gemma (process_prompt)…")
            response_data = self._ask_gemma(final_prompt)

            data_payload = self.extract_data(response_data)
            if data_payload is not None:
                self.logger.info("Gemma returned valid JSON payload with 'Companies'")
                return data_payload

            msg = response_data.get("message", "Unknown error")
            self.logger.error(f"Gemma processing failed or returned no data: {msg}")
            return None

        except Exception as e:
            self.logger.error(f"process_prompt: Unexpected exception: {e}")
            return None

    def process_prompts(self, tasks: list[tuple[str, str]]) -> list[Optional[Dict[str, Any]]]:
        """
        Send multiple prompts in parallel (up to GEMMA_NUM_THREADS concurrent calls).

        Args:
            tasks (List[(template_path, custom_input)]):
                Each tuple is (path_to_template, the_input_string).
        Returns:
            List[Optional[Dict]]: Result list in the same order as `tasks`. Each element
                                  is the parsed JSON dict from Gemma or None on failure.
        """
        results: list[Optional[Dict[str, Any]]] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=GEMMA_NUM_THREADS) as executor:
            future_to_idx = {}
            for idx, (template_path, custom_input) in enumerate(tasks):
                future = executor.submit(self.process_prompt, template_path, custom_input)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"process_prompts: Task {idx} raised exception: {e}")
                    results[idx] = None

        return results