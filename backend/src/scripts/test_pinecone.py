import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config.settings import settings

print(f"Pinecone API Key set: {bool(settings.pinecone_api_key)}")
print(f"Pinecone Index Name: {settings.pinecone_index_name}")
print(f"OpenAI API Key set: {bool(settings.openai_api_key)}")

if settings.pinecone_api_key:
    print(f"\nAPI Key (first 10 chars): {settings.pinecone_api_key[:10]}...")
    
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        print("\n✓ Pinecone client initialized successfully")
        
        # Try to get the index
        try:
            index = pc.Index(settings.pinecone_index_name)
            print(f"✓ Successfully connected to index: {settings.pinecone_index_name}")
            
            # Get index stats
            stats = index.describe_index_stats()
            print(f"✓ Index stats: {stats}")
        except Exception as e:
            print(f"✗ Error accessing index '{settings.pinecone_index_name}': {e}")
            print(f"\nPlease ensure the index exists in your Pinecone console.")
            
    except Exception as e:
        print(f"\n✗ Error initializing Pinecone client: {e}")
else:
    print("\n✗ PINECONE_API_KEY is not set in your .env file")
