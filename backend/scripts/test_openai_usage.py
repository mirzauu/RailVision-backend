import asyncio
import os
import sys
import time

# Add backend to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.infrastructure.llm.provider_service import ProviderService
from src.api.v1.provider.schemas import UsageCostRequest

async def main():
    try:
        service = ProviderService(user_id="test-user")
        
        # Check if API key is available
        api_key = service._get_api_key("openai")
        if not api_key:
            print("OPENAI_API_KEY not found in environment or settings.")
        else:
            print(f"Testing OpenAI Usage API with key: {api_key[:5]}...")
        
        # Request for last 30 days
        end_time = int(time.time())
        start_time = end_time - (30 * 24 * 60 * 60)
        
        request = UsageCostRequest(
            start_time=start_time,
            end_time=end_time,
            limit=10,
            bucket_width="1d"
        )
        
        print(f"Requesting usage from {start_time} to {end_time}")
        response = await service.get_openai_usage_costs(request)
        print("Response received:")
        print(response.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
