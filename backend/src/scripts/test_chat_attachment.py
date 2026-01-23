"""
Test script for chat with file attachment.
This script tests the /chat/stream/upload endpoint.
"""
import requests
import os

BASE_URL = "http://localhost:8000/api/v1"

# You'll need to set these values
TOKEN = os.getenv("TEST_TOKEN", "")  # JWT token for authentication
PROJECT_ID = os.getenv("TEST_PROJECT_ID", "default")

# Test file path (update this to test with your own file)
TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test_val.docx")


def test_chat_with_attachment():
    """Test the chat endpoint with a file attachment."""
    
    if not TOKEN:
        print("ERROR: Please set TEST_TOKEN environment variable with a valid JWT token")
        print("You can get a token by logging in via the /auth/login endpoint")
        return
    
    if not os.path.exists(TEST_FILE_PATH):
        print(f"ERROR: Test file not found: {TEST_FILE_PATH}")
        return
    
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }
    
    # Prepare multipart form data
    with open(TEST_FILE_PATH, "rb") as f:
        files = {
            "file": (os.path.basename(TEST_FILE_PATH), f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        data = {
            "query": "What is this document about? Summarize the key points.",
            "project_id": PROJECT_ID,
            "framework": "pydantic",
        }
        
        print(f"Sending request to {BASE_URL}/conversations/chat/stream/upload")
        print(f"File: {TEST_FILE_PATH}")
        print(f"Query: {data['query']}")
        print("-" * 50)
        
        response = requests.post(
            f"{BASE_URL}/conversations/chat/stream/upload",
            headers=headers,
            data=data,
            files=files,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"ERROR: Request failed with status {response.status_code}")
            print(response.text)
            return
        
        print("Response (streaming):")
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))


if __name__ == "__main__":
    test_chat_with_attachment()
