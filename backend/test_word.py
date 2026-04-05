import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv("../.env")

from src.infrastructure.agents.tools.word_tool import _run_claude_sync

prompt = """
Generate a professional Word document titled 'Renewable Energy Report'.
Use python-docx to generate it.
"""

output_path = "storage/word_docs/test_word_output.docx"
print(f"Testing Word Generation directly using word_tool._run_claude_sync...")
print(f"Output will be saved to: {output_path}")

success, text = _run_claude_sync(prompt, output_path)

if success:
    print(f"\n✅ Test passed! File size: {os.path.getsize(output_path)} bytes")
else:
    print("\n❌ Test failed!")
