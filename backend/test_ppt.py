import sys
import os

# Add src to path just in case
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv("../.env")

from src.infrastructure.agents.tools.ppt_tool import _run_claude_sync

prompt = """
Generate a highly professional PowerPoint presentation titled 'AI Innovations'.
Use python-pptx to generate the presentation, save it, and ensure it creates the file output perfectly.
Slide 1:
Slide Type: title
Title: AI Innovations
Content: Reimagining the future.

Slide 2:
Slide Type: bullet
Title: Key Features
Content:
- Advanced AI
- Fast execution
- Custom formatting
"""

output_path = "storage/presentations/test_claude_sync_output.pptx"
print(f"Testing PPT Generation directly using ppt_tool._run_claude_sync...")
print(f"Output will be saved to: {output_path}")

success = _run_claude_sync(prompt, output_path)

if success:
    print(f"\n✅ Test passed! File completely generated and downloaded at: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
else:
    print("\n❌ Test failed! Could not generate or capture file.")
