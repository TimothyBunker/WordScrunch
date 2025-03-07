import json
import nltk
import ollama
import multiprocessing
from nltk.corpus import words


# LOAD STUFF
nltk.download('words', quiet=True)
english_words = words.words()
model_name = "deepseek-r1:7b"


def generate_descriptions(word, num_sentences=5):
    try:
        prompt = f"Provide {num_sentences} different sentences that describe the word '{word}':"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        full_response = response['message']['content'].strip()

        if "<think>" in full_response:
            full_response = full_response.split("</think>")[-1].strip()

        descriptions = full_response.split('\n')
        return word, [desc.strip() for desc in descriptions if desc.strip()]

    except Exception as e:
        print(f"Error generating descriptions for '{word}': {e}")
        return word, []


def process_words(word_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1) as pool:
        results = pool.map(generate_descriptions, word_list)
    return dict(results)

if __name__ == "__main__":
    batch_size = 500
    total_words = len(english_words)

    try:
        with open("../word_descriptions.json", "r", encoding="utf-8") as file:
            word_descriptions = json.load(file)
            print(f"Resuming with {len(word_descriptions)} words already processed.")
    except FileNotFoundError:
        print("No previous save found. Starting fresh.")
        word_descriptions = {}

    processed_count = len(word_descriptions)
    for i in range(0, total_words, batch_size):
        batch = english_words[i:i + batch_size]

        batch = [word for word in batch if word not in word_descriptions]

        if not batch:
            continue

        print(f"Processing batch {i // batch_size + 1}...")

        batch_descriptions = process_words(batch)
        word_descriptions.update(batch_descriptions)

        with open("../word_descriptions.json", "w", encoding="utf-8") as file:
            json.dump(word_descriptions, file, ensure_ascii=False, indent=2)

        processed_count += len(batch)
        print(f"Saved progress: {processed_count}/{total_words} words processed.")

    print("Completed processing all words.")
