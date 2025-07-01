import httpx  # The async-capable replacement for 'requests'
import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any

# Define the Ollama API endpoint
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


# --- ASYNCHRONOUS FUNCTIONS ---

async def ask_ollama_async(prompt: str, model: str = "llama3.1:8b") -> Optional[Dict[str, Any]]:
    """
    Asynchronously sends a prompt to the Ollama API and gets the full response.

    Args:
        prompt: The input text to send to the model.
        model: The name of the Ollama model to use.

    Returns:
        The full JSON response dictionary from Ollama, or None if an error occurs.
        Note: The previous optimizer script expects the full dictionary.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        # httpx.AsyncClient is the async equivalent of a requests.Session
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 'await' pauses this function until the network request is complete
            response = await client.post(OLLAMA_ENDPOINT, json=payload)

            # Check for HTTP errors (e.g., 404, 500)
            response.raise_for_status()

            # .json() is also an awaitable coroutine
            return await response.json()

    except httpx.RequestError as e:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_ENDPOINT}.")
        print(f"Details: {e}")
        print("Please make sure Ollama is running and the model is available.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during the async Ollama request: {e}")
        return None


async def ask_ollama_stream_async(prompt: str, model: str = "llama3.1:8b") -> AsyncGenerator[str, None]:
    """
    Asynchronously sends a prompt and streams the response token by token.

    Args:
        prompt: The input text to send to the model.
        model: The name of the Ollama model to use.

    Yields:
        Each chunk of the response text as it is generated.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # client.stream() opens a connection without waiting for the full response
            async with client.stream("POST", OLLAMA_ENDPOINT, json=payload) as response:
                response.raise_for_status()

                # 'async for' iterates over the response as chunks arrive
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break

    except httpx.RequestError as e:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_ENDPOINT}.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the async Ollama stream: {e}")

