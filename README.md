# Semantic Entropy Detection

Semantic Entropy calculator and hallucination detector for Mission 10 Conversational AI.

## Overview

This project implements semantic entropy detection to identify when AI models are hallucinating or uncertain about their responses. By generating multiple responses to the same prompt and analyzing their semantic similarity, we can detect whether the model is confidently consistent or making things up.

## Files

- `semantic_entropy.py` - The main semantic entropy calculator, including demo scenarios
- `requirements.txt` - The dependency list of required libraries
- `.gitignore` - Git ignore configuration

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the demo script:

```bash
python semantic_entropy.py
```

The script will:
1. Ask if you have an OpenAI API key (optional)
2. Run two demo scenarios comparing consistent vs. hallucinating responses
3. Display semantic similarity analysis and entropy categorization

### Using in Your Own Code

```python
from semantic_entropy import generate_samples, calculate_semantic_consistency, analyze_uncertainty

# Option 1: Generate responses automatically with OpenAI API
responses = generate_samples(
    prompt="What do the Keplerians dream about?",
    k=5,
    use_api=True,
    api_key="your-api-key-here"
)

# Option 2: Use your own responses
responses = [
    "Response 1 from your model",
    "Response 2 from your model",
    "Response 3 from your model",
    # ... more responses
]

# Analyze semantic consistency
result = calculate_semantic_consistency(responses)

# Quick uncertainty check
is_uncertain = analyze_uncertainty(responses, threshold=0.50)
if is_uncertain:
    print("⚠️ Model is likely hallucinating - don't trust this output!")
```

## How It Works

1. **Generate Multiple Responses**: Ask the same question multiple times with high temperature
2. **Compute Embeddings**: Convert responses to semantic vectors using sentence transformers
3. **Calculate Similarity**: Measure pairwise cosine similarity between all responses
4. **Assess Entropy**: 
   - **LOW entropy** (high similarity) = Model is confident and consistent
   - **HIGH entropy** (low similarity) = Model is uncertain or hallucinating

## Entropy Categories

- **LOW** (score > 0.75): Model is confident, responses are semantically similar
- **MEDIUM** (score 0.50-0.75): Some uncertainty, verify specific claims
- **HIGH** (score < 0.50): Likely hallucinating, don't trust the output

## The First Contact Scenario

This project is part of a fictional scenario where the Starship Dialectic has encountered an alien civilization (the Keplerians) on Kepler-186f. The AI assistant AURA must be tested for hallucination and bias before being authorized for first contact communications.

## Environment Variables

Optionally set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```
