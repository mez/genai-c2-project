
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')  # reads variables from a .env file and sets them in os.environ


def get_openai_api_key():
    """Fetch OpenAI API key from environment or fallback."""
    return os.getenv("OPENAI_API_KEY", "your-api-key-here")

def get_openai_client():
    # Set your API key (get from environment or replace with your key)
    api_key = get_openai_api_key()

    # Initialize the client
    # For Vocareum keys, use: base_url="https://openai.vocareum.com/v1"
    client = OpenAI(
    api_key=api_key,
    base_url="https://openai.vocareum.com/v1" if api_key.startswith("voc") else None
    )

    print("✅ OpenAI client initialized!")
    return client

if __name__ == "__main__":
    print("Testing OpenAI API key...")
    client = get_openai_client()
    # Print client info for debugging
    print("Client info:")
    print("  base_url:", getattr(client, 'base_url', None))
    print("  api_key:", getattr(client, 'api_key', None))
    print("  organization:", getattr(client, 'organization', None))
    try:
        # Make a simple API call to check key validity
        response = client.embeddings.create(input=["test"], model="text-embedding-3-small")
        print("✅ API key is valid! Embedding length:", len(response.data[0].embedding))
    except Exception as e:
        print("❌ API key test failed:", e)
