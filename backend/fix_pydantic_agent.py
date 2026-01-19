import os

file_path = r'd:\UPWORK CLIENT\RailVision backend\backend\src\infrastructure\agents\pydantic_agent.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'provider = llm_provider.chat_config.provider' in line:
        indent = line[:line.find('provider')]
        new_lines.append(f'{indent}provider_config = llm_provider.chat_config\n')
        new_lines.append(f'{indent}provider = provider_config.provider\n')
        new_lines.append(f'{indent}auth_provider = provider_config.auth_provider\n')
        new_lines.append(f'{indent}api_key = llm_provider._get_api_key(auth_provider)\n')
        new_lines.append(f'{indent}base_url = provider_config.base_url\n')
    elif 'model_id = llm_provider.chat_config.model' in line:
        indent = line[:line.find('model_id')]
        new_lines.append(f'{indent}model_id = provider_config.model.split("/")[-1]\n')
    elif 'model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key))' in line:
        indent = line[:line.find('model')]
        new_lines.append(f'{indent}model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url))\n')
    elif 'model = AnthropicModel(model_name=model_id, provider=AnthropicProvider(api_key=api_key))' in line:
        indent = line[:line.find('model')]
        new_lines.append(f'{indent}model = AnthropicModel(model_name=model_id, provider=AnthropicProvider(api_key=api_key, base_url=base_url))\n')
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Replacement complete.")
