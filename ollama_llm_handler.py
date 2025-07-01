import requests
import json

# Define the Ollama API endpoint
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


def ask_ollama(prompt: str, model: str = "gemma3:1b"):
    """
    Sends a prompt to the Ollama API and gets the full response at once.

    Args:
        prompt: The input text to send to the model.
        model: The name of the Ollama model to use (e.g., 'llama3', 'mistral').

    Returns:
        The generated response text as a string, or None if an error occurs.
    """
    # print(f"--- Sending prompt to model: {model} (non-streaming) ---")
    try:
        # The payload to send to the API
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False  # We want the full response at once
        }

        # Send the POST request
        response = requests.post(OLLAMA_ENDPOINT, json=payload)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the JSON response and return the 'response' field
        response_data = response.json()
        return response_data.get("response")

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_ENDPOINT}.")
        print(f"Details: {e}")
        print("Please make sure Ollama is running.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def ask_ollama_stream_response(prompt: str, model: str = "gemma3:1b"):
    """
    Sends a prompt and streams the response token by token.

    Args:
        prompt: The input text to send to the model.
        model: The name of the Ollama model to use.

    Yields:
        Each chunk of the response text as it is generated.
    """
    print(f"\n--- Sending prompt to model: {model} (streaming) ---")
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True  # Enable streaming
        }

        # The 'stream=True' argument in requests.post is crucial for streaming
        with requests.post(OLLAMA_ENDPOINT, json=payload, stream=True) as response:
            response.raise_for_status()

            # Iterate over the response line by line
            for line in response.iter_lines():
                if line:
                    # Each line is a JSON object; decode and parse it
                    chunk = json.loads(line.decode('utf-8'))

                    # Yield the 'response' part of the chunk
                    yield chunk.get("response", "")

                    # The final chunk in a stream has a 'done' key
                    if chunk.get("done"):
                        break

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_ENDPOINT}.")
        print(f"Details: {e}")
        print("Please make sure Ollama is running.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
