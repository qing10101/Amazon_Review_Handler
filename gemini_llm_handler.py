import os
import logging
import asyncio
from typing import Optional

# Use aiofiles for non-blocking file operations
import aiofiles

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_dir = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY_FILE = os.path.join(base_dir, 'key.txt')

# --- Google Gemini Library Import ---
try:
    from google import genai
    from google.api_core import exceptions as google_api_exceptions
except ImportError:
    print("Error: The 'google-genai' library is not installed.")
    print("Please install it using: pip install google-genai")
    genai = None  # Sentinel value
    google_api_exceptions = None

# --- Lazy Initialization Globals ---
# These will be populated by the async initializer on the first call.
GEMINI_CLIENT= None
MODEL_NAME = "gemini-2.5-flash"  # It's good practice to specify a version


# --- ASYNCHRONOUS FUNCTIONS ---

async def get_gemini_key_async(filepath: str = GEMINI_API_KEY_FILE) -> Optional[str]:
    """Asynchronously reads the Gemini API key from a file without blocking."""
    try:
        async with aiofiles.open(filepath, 'r') as f:
            key = await f.read()
            key = key.strip()
            if key:
                return key
            else:
                logging.error(f"Gemini API key file '{filepath}' is empty.")
                return None
    except FileNotFoundError:
        logging.error(f"Gemini API key file '{filepath}' not found. Please create it.")
        return None
    except Exception as e:
        logging.error(f"Error reading Gemini API key file '{filepath}': {e}")
        return None


async def initialize_gemini_client_async():
    """
    Initializes the Gemini client. This function is designed to be called
    once and caches the result in the global GEMINI_CLIENT.
    """
    global GEMINI_CLIENT
    if GEMINI_CLIENT:
        return True  # Already initialized

    if not genai:
        logging.error("Gemini library not found. Gemini features are disabled.")
        return False

    api_key = await get_gemini_key_async()
    if not api_key:
        logging.error("Gemini API key not found or empty. Gemini features will be disabled.")
        return False

    try:
        # The client initialization itself is synchronous and fast.
        GEMINI_CLIENT = genai.Client(api_key=api_key)
        logging.info(f"Successfully initialized Gemini client for model '{MODEL_NAME}'.")
        return True
    except Exception as e:
        logging.error(f"Failed to configure Gemini client: {e}")
        return False


def _synchronous_api_call(prompt: str):
    """
    A private helper function that wraps the blocking API call.
    This is what will be run in a separate thread.
    """
    if GEMINI_CLIENT:
        # This is the original, blocking call.
        return GEMINI_CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    else:
        return None


async def ask_gemini_async(prompt: str) -> Optional[str]:
    """
    Asynchronously asks Gemini a question using a synchronous client.

    This function uses `asyncio.to_thread` to run the blocking network call
    in a separate thread, preventing it from blocking the main event loop.

    Args:
        prompt: The input text to send to the model.

    Returns:
        The generated response text as a string, or None if an error occurs.
    """
    # Ensure the client is initialized before proceeding.
    if not await initialize_gemini_client_async():
        return None  # Initialization failed

    try:
        # Run the synchronous function (_synchronous_api_call) in a separate thread.
        # await pauses this function until the thread completes, but the
        # asyncio event loop is free to run other tasks.
        response = await asyncio.to_thread(_synchronous_api_call, prompt)

        if not response or not response.text:
            logging.warning("Gemini response was empty.")
            return None

        return response.text.strip()

    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Gemini API error: {e}")
        return None
    except ValueError as e:
        logging.error(f"Gemini value error (potentially blocked prompt): {e}")
        return None
    except Exception as e:
        # This will catch errors from both the thread execution and this function.
        logging.error(f"An unexpected error occurred during Gemini request: {type(e).__name__} - {e}")
        return None


# --- EXAMPLE USAGE BLOCK ---
async def main():
    """Example function to demonstrate and test the async handler."""
    print("--- Testing ask_gemini_async (with legacy API call in a thread) ---")
    response = await ask_gemini_async("What is the main benefit of using asyncio.to_thread?")

    if response:
        print("Response received:\n", response)
    else:
        print("Failed to get a response. Check logs for errors.")


if __name__ == "__main__":
    asyncio.run(main())