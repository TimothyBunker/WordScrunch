import json
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


def clean_descriptions(raw_data):
    """
    For each word key in the raw JSON data, extract the actual descriptions.
    Assumes that any line that follows a number in succession is part of a description.
    Joins the valid description lines into a single string.
    """
    cleaned_data = {}
    for word, desc_list in raw_data.items():
        clean_descs = []

        # Skip lines that are headers/numbering and too short.
        for line in desc_list:
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^\*\*.*?\*\*:", "", line)
            line = line.strip()
            if len(line) < 5 or line.lower().startswith("here are"):
                continue
            clean_descs.append(line)
        if clean_descs:
            cleaned_data[word] = " ".join(clean_descs)
    return cleaned_data

# Load raw data from JSON (adjust file path as needed)
with open("../word_descriptions.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = clean_descriptions(raw_data)
print(f"Cleaned descriptions for {len(cleaned_data)} words.")


dataset = [{"word": word, "description": desc} for word, desc in cleaned_data.items()]
word_to_index = {entry["word"]: idx for idx, entry in enumerate(dataset)}
vocab_size = len(dataset)
print(f"Dataset size (vocabulary): {vocab_size}")


# Start our embeddings in a space that is pretty uniform so the model is at least a little normalized to start
class UniformEmbeddingSpace(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        embeddings = torch.randn(vocab_size, embedding_dim)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        self.embeddings = nn.Parameter(embeddings)  # Trainable

    def forward(self, token_ids):
        return self.embeddings[token_ids]

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

def get_attention_scores(word, description):
    """Extracts attention scores from BERT for the given description and word."""
    inputs = bert_tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1]  # Last layer's attention
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_tokens = bert_tokenizer.tokenize(word)
    word_indices = [i for i, token in enumerate(tokens) if token in word_tokens]
    if not word_indices:
        return None
    word_attention = attention[:, :, word_indices, :].mean(dim=1)  # [1, num_target_tokens, seq_len]
    return tokens, word_attention

def compute_context_embedding(tokens, attention_scores, embedding_space, bert_tokenizer):
    """
    Computes an attention-weighted context embedding.
    Returns c_t of shape [1, embedding_dim].
    """
    token_ids = [bert_tokenizer.convert_tokens_to_ids(t) for t in tokens]
    token_ids_tensor = torch.tensor(token_ids)
    context_embeddings = embedding_space(token_ids_tensor)  # [num_tokens, embedding_dim]
    # Average over target tokens axis: from [1, num_target_tokens, seq_len] -> [1, seq_len]
    attn_vals = attention_scores.mean(dim=1)  # [1, seq_len]
    if attn_vals.shape[1] != len(tokens):
        print("Warning: Number of attention scores does not match number of tokens.")
    attn_vals = attn_vals.squeeze(0)  # [seq_len]
    attention_values = attn_vals.unsqueeze(1)  # [seq_len, 1]
    c_t = torch.sum(attention_values * context_embeddings, dim=0, keepdim=True)  # [1, embedding_dim]
    return c_t

class CloserEmbeddingUpdater(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2 + 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, embedding_dim)
        self.gate = nn.Linear(embedding_dim * 2 + 1, 1)  # Scalar gate

    def forward(self, e_t, c_t, A):
        """
        e_t: Target word embedding [1, embedding_dim]
        c_t: Context embedding [1, embedding_dim]
        A:   Attention score scalar [1, 1]
        """
        x = torch.cat([e_t, c_t, A], dim=-1)  # Now all are [1, ...]
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        delta_e = self.fc3(h)
        g = torch.sigmoid(self.gate(x))
        delta_e = torch.where((c_t - e_t) * delta_e > 0, delta_e, torch.zeros_like(delta_e))
        e_new = e_t + g * delta_e
        return e_new


# for embedding our embeddings in a closer space, awww :D
def contrastive_loss(e_t, c_t, neg_e, A, margin=1.0):
    """
    Computes contrastive loss with negative sampling.
    """
    positive_loss = A * (1 - F.cosine_similarity(e_t, c_t))
    negative_loss = (1 - A) * F.relu(margin - F.cosine_similarity(e_t, neg_e))
    return (positive_loss + negative_loss).mean()


embedding_dim = 768
embedding_space = UniformEmbeddingSpace(vocab_size, embedding_dim)
embedding_updater = CloserEmbeddingUpdater(embedding_dim)
optimizer = optim.Adam(embedding_updater.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    processed = 0
    for entry in dataset:
        word = entry["word"]
        desc = entry["description"]

        # Get target word embedding.
        word_id = word_to_index[word]
        e_t = embedding_space(torch.tensor([word_id]))

        # Get attention scores from BERT.
        result = get_attention_scores(word, desc)
        if result is None:
            continue
        tokens, attn_scores = result

        # Compute context embedding.
        c_t = compute_context_embedding(tokens, attn_scores, embedding_space, bert_tokenizer)
        if c_t is None:
            continue

        # Sample a negative word embedding.
        neg_word_id = random.choice(list(word_to_index.values()))
        neg_e = embedding_space(torch.tensor([neg_word_id]))

        # Reduce the attention scores to a scalar.
        # Instead of attn_scores.mean(dim=0, keepdim=True) (which may be 3D),
        # we take the mean over all dimensions and reshape to [1, 1].
        A = attn_scores.mean().unsqueeze(0).unsqueeze(0)  # Now A is [1,1]

        optimizer.zero_grad()
        e_new = embedding_updater(e_t, c_t, A)
        loss = contrastive_loss(e_new, c_t, neg_e, A)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        processed += 1

    print(f"Epoch {epoch + 1}: Processed {processed} entries, Loss = {total_loss:.4f}")
