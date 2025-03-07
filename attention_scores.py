import json
import re
import random
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


'''
ChatGPTs attempt at formatting all the attention scores and embeddings.
We'll use this to precompute the attention scores and embeddings for all words in the dataset.
We'll use this to train our word embeddings.
Didn't feel like doing the manual labor of that just yet

will get around to it, if it sucks too much, I fear it might sadly.
'''
with open("../cleaned_word_descriptions.json", "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

print(f"Loaded cleaned data for {len(cleaned_data)} words.")

word_to_index = {word: idx for idx, word in enumerate(cleaned_data.keys())}
vocab_size = len(word_to_index)
embedding_dim = 768

class UniformEmbeddingSpace(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Initialize embeddings randomly and normalize to lie on a unit hypersphere.
        embeddings = torch.randn(vocab_size, embedding_dim)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        self.embeddings = nn.Parameter(embeddings)  # Trainable parameters

    def forward(self, token_ids):
        return self.embeddings[token_ids]


embedding_space = UniformEmbeddingSpace(vocab_size, embedding_dim)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)


def get_attention_scores(word, description):
    """
    For a given word and its description text, returns:
      - tokens: list of tokens from BERT tokenization of the description.
      - attn_scores: a tensor of shape [1, num_target_tokens, seq_len] containing the attention scores,
        where num_target_tokens corresponds to the positions in the description where any token of the word appears.
    """
    inputs = bert_tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1]  # Use last layer's attention (shape: [batch, num_heads, seq_len, seq_len])
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Tokenize the target word using BERT tokenizer.
    word_tokens = bert_tokenizer.tokenize(word)
    # Find all indices where any token from word_tokens appears.
    word_indices = [i for i, token in enumerate(tokens) if token in word_tokens]
    if not word_indices:
        return None
    # Aggregate attention scores over the found indices (average over the target token dimension).
    # This gives us a tensor of shape [1, num_target_tokens, seq_len].
    attn_scores = attention[:, :, word_indices, :].mean(dim=1)
    return tokens, attn_scores


##########################################
# 3. Precompute Attention Scores and Initial Embeddings
##########################################
precomputed_data = {}
num_processed = 0

for word, description in cleaned_data.items():
    result = get_attention_scores(word, description)
    if result is None:
        continue
    tokens, attn_scores = result
    # Convert attention scores to list.
    # attn_scores is of shape [1, num_target_tokens, seq_len]. We'll squeeze the batch dimension.
    attn_scores_list = attn_scores.squeeze(0).tolist()

    # Retrieve the initial embedding for the word from our uniform embedding space.
    # Our word_to_index mapping gives a unique index per word.
    word_id = word_to_index[word]
    embedding_tensor = embedding_space(torch.tensor([word_id]))
    # Convert the embedding tensor to a list of floats.
    embedding_list = embedding_tensor.squeeze(0).tolist()

    # Save all precomputed information under this word.
    precomputed_data[word] = {
        "description": description,
        "tokens": tokens,
        "attention_scores": attn_scores_list,
        "embedding": embedding_list
    }
    num_processed += 1

print(f"Precomputed data for {num_processed} words.")

# Save the precomputed data to a JSON file.
with open("../precomputed_attention_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(precomputed_data, f, indent=4)

print("Precomputed attention scores and embeddings saved to 'precomputed_attention_embeddings.json'.")
