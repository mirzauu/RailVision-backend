import os

file_path = r'd:\UPWORK CLIENT\RailVision backend\backend\src\infrastructure\agents\pydantic_agent.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i in range(len(lines)):
    line = lines[i]
    # Remove the duplicate api_key assignment
    if 'api_key = llm_provider._get_api_key(llm_provider.chat_config.auth_provider)' in line:
        continue
    # Ensure redundant OpenAI fallback is cleaned up if we want, but it's okay to keep if safe.
    # Actually, let's keep it simple and just fix the basics.
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Cleanup complete.")
