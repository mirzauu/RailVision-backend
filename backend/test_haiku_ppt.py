import sys
import os
import re

# Add src to path just in case
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv("../.env")

prompt = """
Generate a PowerPoint presentation about RailVision matching the RailVision theme.
"""

def _run_claude_sync_verbose(prompt: str, storage_rel: str) -> bool:
    try:
        from anthropic import Anthropic
        api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)
        
        all_file_ids = []

        with client.beta.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            betas=["code-execution-2025-08-25", "skills-2025-10-02", "files-api-2025-04-14"],
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            extra_body={
                "container": {
                    "skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]
                }
            }
        ) as stream:
            for event in stream:
                event_str = str(event)
                print("DEBUG EVENT:", event_str[:500] if len(event_str) > 500 else event_str) # PRINT EVENTS for debugging
                
                # Match file_ followed by at least 15 alphanumeric characters
                matches = re.findall(r'file_[a-zA-Z0-9]{15,}', event_str)
                if matches:
                    for fid in matches:
                        if fid not in all_file_ids:
                            all_file_ids.append(fid)

        print("ALL FOUND FILE IDS:", all_file_ids)
        if all_file_ids:
            return True
        return False
    except Exception as e:
        print("EXCEPTION:", e)
        return False

output_path = "storage/presentations/test_haiku_output.pptx"
print(f"Testing PPT Generation using Haiku...")
success = _run_claude_sync_verbose(prompt, output_path)

if success:
    print(f"\n✅ Test passed!")
else:
    print("\n❌ Test failed!")
