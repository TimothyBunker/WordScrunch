# WordScrunch - Attention-Guided Uniform Embedding Learning

![Clustered Embedding](/Clustered.png)

## Overview

This project aims to create a novel embedding space for words using attention signals extracted from BERT. Instead of relying on pre-trained embeddings (like those from Word2Vec or BERT) tuned for prediction tasks, we start with uniformly distributed, maximally distant embeddings and update them solely based on context. The key idea is to use precomputed attention scores from BERT, derived from word descriptions, to guide a contrastive loss update—ensuring that embeddings only move closer when contextually justified.

## Achievements

- **Cleaned Data Pipeline:**  
  Developed a cleaning script to extract and consolidate word descriptions from a raw JSON file into a uniform format.

- **Uniform Embedding Initialization:**  
  Implemented a custom embedding space that initializes embeddings on a unit hypersphere so they start maximally distant from each other.

- **Attention Score Extraction:**  
  Leveraged BERT to compute attention scores from the description of each word. These scores indicate how much each token in a description is relevant to the target word.

- **Precomputation and Storage:**  
  Precomputed attention scores (along with BERT tokenization) for each word and saved them together with the corresponding initial embedding. This avoids recomputation during training and enables multi-level aggregation in the future.

- **Contrastive Learning with Negative Sampling:**  
  Designed an update mechanism using a gated MLP and a contrastive loss function. This mechanism updates only the target word embeddings by moving them closer to their attention-weighted context embeddings (and pushing apart negative samples) without disrupting the overall uniform structure.

## Current Pipeline

1. **Data Preparation:**  
   - A cleaned JSON file (`cleaned_word_descriptions.json`) maps each word to its description.
   - A precomputation script processes each entry with BERT to obtain tokenized descriptions and corresponding attention scores.

2. **Embedding Initialization:**  
   - A custom `UniformEmbeddingSpace` module creates trainable embeddings that are uniformly distributed (maximally distant) at the start.

3. **Attention-Based Updates:**  
   - BERT extracts attention scores using `get_attention_scores()`.
   - The `compute_context_embedding()` function computes an attention-weighted sum of token embeddings.
   - The `CloserEmbeddingUpdater` model uses a gated MLP to update target embeddings based on the context.

4. **Training with Contrastive Loss:**  
   - A contrastive loss with negative sampling ensures that embeddings move closer to their positive (context) embeddings while staying distinct from negative samples.
   - The training loop updates embeddings iteratively based on precomputed attention signals.

## Future Direction

- **Enhanced Aggregation:**  
  Explore multi-level (tertiary) aggregation of attention scores from multiple descriptions for each word to further refine the update signal.

- **Hierarchical and Adaptive Updates:**  
  Investigate differential update speeds (e.g., based on word frequency or part-of-speech) and additional constraints to prevent over-collapsing of the embedding space.

- **Integration with Language Models:**  
  Aim to use the updated embeddings as a component in larger models (e.g., GPT-2) to test their effectiveness in downstream language tasks.

- **Visualization and Analysis:**  
  Generate t-SNE or UMAP plots to visualize the evolution of the embedding space over time and validate the semantic clustering.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- JSON (standard library)
- Other dependencies as needed (e.g., regex)

## Important Notes

- I am still generating descriptions for all the words so everything up until this point is still in progress
- I am doing so with a deepseek model (deepseek-r1:32b)
- For simplicity’s sake I have been doing this on a 4090 but if that's too slow I will look into cloud compute services

## Setup & Running

1. **Install Dependencies:**
   ```bash
   pip install torch transformers
