import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the project root to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables
load_dotenv()

from src.infrastructure.llm.provider_service import ProviderService

async def test_claude():
    print("Testing Claude Integration...")
    
    # Check if API key is present in env
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("Error: CLAUDE_API_KEY not found in environment variables.")
        return

    print(f"CLAUDE_API_KEY found: {api_key[:10]}...")

    service = ProviderService(user_id="test-user")
    
    models_to_test = [
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-sonnet-20240620",
        "anthropic/claude-3-5-sonnet-latest",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-haiku-20240307"
    ]

    for model in models_to_test:
        print(f"\nTesting model: {model}")
        messages = [{"role": "user", "content": "Hello, are you Claude? Reply with 'Yes'."}]
        
        try:
            print(f"Sending request to {model}...")
            response = await service.call_llm_with_specific_model(
                model_identifier=model,
                messages=messages
            )
            print("--- Response ---")
            print(response)
            print("----------------")
            print(f"Test PASSED for {model}")
            break # Stop after first success
        except Exception as e:
            print(f"Test FAILED for {model}: {e}")

if __name__ == "__main__":
    asyncio.run(test_claude())
