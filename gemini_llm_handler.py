import os
import logging


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_dir = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY_FILE = os.path.join(base_dir,'key.txt')  # Use a separate file for Gemini key


# --- Gemini API Setup ---
def get_gemini_key(filepath=GEMINI_API_KEY_FILE):
    """Reads the Gemini API key from a file."""
    try:
        with open(filepath, 'r') as f:
            key = f.read().strip()
            if key:
                return key
            else:
                logging.error(f"Gemini API key file '{filepath}' is empty.")
                return None
    except FileNotFoundError:
        logging.error(f"Gemini API key file '{filepath}' not found. Create it and add your key.")
        return None
    except Exception as e:
        logging.error(f"Error reading Gemini API key file '{filepath}': {e}")
        return None

# --- Google Gemini ---
try:
    from google import genai
    from google.api_core import exceptions as google_api_exceptions
except ImportError:
    print("Error: The 'google-genai' library is not installed.")
    print("Please install it using: pip install google-genai")
    genai = None # Set genai to None if import fails

# Configure Gemini client
gemini_api_key = get_gemini_key()
client = genai.Client(api_key=get_gemini_key())

if client and gemini_api_key:
    model = "gemini-2.5-flash"
elif not genai:
    logging.error("Gemini library not found. Gemini features will be disabled.")
else:
    logging.error("Gemini API key not found or empty. Gemini features will be disabled.")


def ask_gemini(prompt: str):
    if not client:  # Check if the client object was successfully created
        return {"error": "Gemini client not initialized. Cannot analyze content."}

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        if response is None or response == "":
            logging.error("Gemini response is empty.")

        response = response.text.strip()
        return response

    # Handle specific Google API errors if needed
    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Gemini API error: {e}")
        return {"error": f"Gemini API error: {e}"}
    except ValueError as e:
        # Catch potential errors like blocked prompts due to safety settings if not handled above
        logging.error(f"Gemini value error (potentially blocked content): {e}")
        return {"error": f"Gemini analysis error (potentially blocked content): {e}"}
    except Exception as e:
        # Catch-all for other unexpected errors
        logging.error(f"Error during Gemini response: {type(e).__name__} - {e}")
        return {"error": f"An unexpected error occurred during Gemini response: {e}"}
