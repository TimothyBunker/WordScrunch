import json
import re

# Load dataset
with open("../word_descriptions.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

cleaned_data = {}

for word, descriptions in dataset.items():
    clean_descs = []

    for desc in descriptions:
        desc = re.sub(r"^\d+\.\s*", "", desc)

        desc = re.sub(r"^\*\*.*?\*\*:", "", desc)

        desc = re.sub(r"\s+([,.!?])", r"\1", desc)

        desc = desc.strip()

        if len(desc) > 3 and not desc.lower().startswith("here are"):
            clean_descs.append(desc)
    if clean_descs:
        cleaned_data[word] = clean_descs

with open("../cleaned_word_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print(f"✅ Cleaned {len(cleaned_data)} words and saved to 'cleaned_word_descriptions.json'")
