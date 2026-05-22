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

## 🧠 BERT → Vector DB → Semantic Retrieval | Interactive Teaching Tool

An interactive, single-file web application designed to visually demonstrate how Natural Language Processing (NLP) models, transformers, and vector databases work together to power semantic search.

This tool is built completely using vanilla web technologies (**HTML5, CSS3, and JavaScript**) with zero external runtime dependencies. It serves as an intuitive sandbox for students, developers, and AI enthusiasts trying to understand modern vector embeddings.

### 🚀 Live Demo & How to Use
Since the entire pipeline is simulated right inside the browser, you don't need to configure a backend server, Python environment, or external databases. 

1. Save the `bert-vectordb-tool.html` file locally.
2. Double-click the file to open it in any modern web browser (Chrome, Firefox, Safari, Edge).
3. Type a sentence in **Step 1** and click **Encode** to watch it travel through the pipeline!

### 🔍 Interactive Pipeline Breakdown

The application guides users through a comprehensive **10-step educational workflow**:

* **1. Sentence Input:** The genesis of raw text ingestion.
* **2. Tokenization:** Visualizes a simulated `WordPiece` tokenizer adding `[CLS]` and `[SEP]` tokens.
* **3. Token Embeddings:** Maps token IDs to continuous dense vectors with color-coded positive/negative weights.
* **4. Q · K · V Attention:** A step-by-step math conceptualizer for Query, Key, and Value matrix multiplications.
* **5. Attention Weights:** Renders a dynamic, interactive attention intensity heatmap.
* **6. CLS Vector:** Demonstrates how sentence pooling isolates the definitive sentence-level embedding.
* **7. Vector DB:** Tracks multi-vector indices and maps them visually using a custom **2D PCA Projection Plot**.
* **8. Query & Search:** Accepts search queries, runs the pipeline, and returns ranked results.
* **9. Similarity Deep Dive:** Explains the step-by-step mechanical arithmetic behind **Cosine Similarity**.
* **10. Semantic Map:** Connects individual word embeddings directly back to core database concepts.

### 🛠️ Technical Implementation Details
* **Tokenizer Simulation:** Replicates deterministic subword splitting (`playing` ➔ `play` + `##ing`) via an explicit internal vocabulary mapping array.
* **Vector Engine:** Processes multi-dimensional matrices natively in JavaScript using deterministic pseudo-random seeds.
* **Dimensionality Reduction:** Implements a localized power-iteration version of Principal Component Analysis (**PCA**) to dynamically render high-dimensional vector distributions onto an HTML5 Canvas coordinate plane.

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
