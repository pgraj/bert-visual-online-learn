# BERT Attention and Vector DB Learning Tool

A small educational project that helps students understand how BERT works internally, how tokenisation splits text into tokens, how attention is represented as matrices, and how sentence embeddings can be stored and searched in a simple vector database.

The project is designed for learning, not production. It uses a small set of sample inputs and visual explanations to make the backend flow easier to understand.

## What this project shows

- How a sentence is tokenised into BERT tokens.
- How tokens are converted into tensors and embeddings.
- How BERT attention can be visualised as heatmaps.
- How the CLS token can be inspected across layers and heads.
- How query, key, and value transformations work in attention.
- How sentence embeddings can be stored in a simple vector database.
- How similarity search works using cosine similarity.
- How the overall pipeline flows from sentence input to retrieval result.

## Project files

- `bert_attention_tool.py`  
  Command-line tool for exploring BERT attention outputs and visualising attention matrices.

- `bert_attention_gui.py`  
  GUI version for interactively entering sentences and viewing attention heatmaps.

- `bert_vectordb_gui_full.py`  
  Full GUI learning tool covering tokenisation, embeddings, QKV flow, CLS vectors, vector DB storage, and similarity search.

## Features

### BERT attention exploration
- Enter a sentence and inspect the resulting tokens.
- Visualise CLS attention for a chosen layer and head.
- Visualise attention for a chosen token.
- Compare a token’s attention across layers.
- Compare a token’s attention across heads.

### Vector DB learning flow
- Encode sentences into BERT CLS embeddings.
- Add encoded sentences to a simple in-memory vector database.
- Search stored vectors using cosine similarity.
- Visualise the database in 2D using PCA.
- Show a step-by-step explanation of the retrieval flow.

### Teaching visuals
- Heatmaps for attention matrices.
- Matrix-style views of token relationships.
- Flow diagrams showing the backend pipeline.
- Step-by-step QKV explanation for learning purposes.

## How it works

1. A sentence is entered by the user.
2. BERT tokenises the sentence into subword tokens.
3. Tokens are converted into input IDs and passed through the model.
4. The model produces embeddings and attention tensors.
5. Attention values are displayed as matrices/heatmaps.
6. The CLS embedding is used as a sentence vector.
7. Sentence vectors can be stored in a simple vector DB.
8. A query sentence is encoded and compared with stored vectors using cosine similarity.

## Backend concepts explained

This project is useful for understanding the following concepts:

- **Tokenisation**: text is split into smaller units before being processed.
- **Embeddings**: each token becomes a vector representation.
- **Tensors**: model inputs and outputs are stored as multi-dimensional arrays.
- **Attention matrices**: token-to-token relationships are represented numerically.
- **CLS embedding**: a single vector representing the whole sentence.
- **Vector search**: similar sentences are found by comparing embedding similarity.

## Requirements

- Python 3.9 or later.
- PyTorch.
- Hugging Face Transformers.
- Matplotlib.
- Seaborn.
- NumPy.
- Scikit-learn.
- Tkinter.

## Installation

```bash
pip install torch transformers matplotlib seaborn numpy scikit-learn
```

Tkinter is included with many Python installations, but on some systems you may need to install it separately.

## Run the project

### Attention CLI
```bash
python bert_attention_tool.py
```

### Attention GUI
```bash
python bert_attention_gui.py
```

### Full vector DB learning GUI
```bash
python bert_vectordb_gui_full.py
```

## Example learning use case

A student can enter a sentence such as:

```text
The cat sat on the mat
```

Then the tool can show:
- The tokenised form of the sentence.
- The attention heatmap for a chosen layer and head.
- How the CLS vector is formed.
- How a query sentence is matched against stored vectors.

## Intended audience

This project is mainly for:
- Students learning BERT.
- People new to transformer models.
- Learners who want a visual explanation of vector databases.
- Anyone who wants to see what happens “behind the scenes” in a transformer pipeline.

## Notes

- The vector database in this project is a simple in-memory example, not a production database.
- The project is intentionally small so the learning flow is easy to follow.
- The visualisations are meant to explain the concepts, not to replace a full training or inference system.

## Contributing

Contributions are welcome if they improve:
- Explanations.
- Visual clarity.
- Teaching value.
- Code readability.
- Sample data for learning.

## License

Add your preferred license here before publishing to GitHub.
