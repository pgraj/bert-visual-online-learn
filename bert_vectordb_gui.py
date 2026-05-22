import tkinter as tk
from tkinter import ttk, messagebox

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel

# -----------------------------
# Global setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
).to(DEVICE)
model.eval()

current_sentence = None
current_tokens = None
current_attentions = None
current_cls_embedding = None

# simple in-memory vector DB: list of (text, embedding)
vector_db = []


# -----------------------------
# Core BERT logic
# -----------------------------
def encode_sentence(sentence: str):
    global current_sentence, current_tokens, current_attentions, current_cls_embedding
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    current_attentions = outputs.attentions
    last_hidden = outputs.last_hidden_state  # (1, seq, 768)
    cls_vec = last_hidden[:, 0, :]          # (1, 768)
    current_cls_embedding = cls_vec[0].detach().cpu().numpy()
    current_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    current_sentence = sentence


# -----------------------------
# Vector DB logic
# -----------------------------
def add_to_vector_db():
    if current_sentence is None or current_cls_embedding is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    vector_db.append((current_sentence, current_cls_embedding.copy()))
    messagebox.showinfo("Vector DB", f"Added to DB:\n{current_sentence}")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def search_vector_db(query_embedding, top_k=5):
    scores = []
    for text, emb in vector_db:
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, text))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]


# -----------------------------
# Visualization helpers
# -----------------------------
def show_heatmap(matrix, xlabels, ylabels, title, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=xlabels,
        yticklabels=ylabels,
        cmap=cmap,
        annot=False,
        cbar=True
    )
    plt.title(title)
    plt.xlabel("Key (attended to)")
    plt.ylabel("Query (attending)")
    plt.tight_layout()
    plt.show()


def show_vector_bar(embedding, title="Embedding"):
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(embedding)), embedding, color="purple")
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Attention visualizations
# -----------------------------
def visualize_cls_attention(layer, head):
    if current_tokens is None or current_attentions is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    try:
        attn = current_attentions[layer][0, head].detach().cpu().numpy()
    except Exception as e:
        messagebox.showerror("Error", f"Invalid layer/head: {e}")
        return
    cls_row = attn[0]
    show_heatmap(
        cls_row[None, :],
        current_tokens,
        ["[CLS]"],
        f"[CLS] attention – Layer {layer}, Head {head}",
        cmap="magma"
    )


def visualize_token_attention(layer, head, token):
    if current_tokens is None or current_attentions is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    if token not in current_tokens:
        messagebox.showerror("Error", f"Token '{token}' not found.\nTokens: {current_tokens}")
        return
    idx = current_tokens.index(token)
    try:
        attn = current_attentions[layer][0, head].detach().cpu().numpy()
    except Exception as e:
        messagebox.showerror("Error", f"Invalid layer/head: {e}")
        return
    row = attn[idx]
    show_heatmap(
        row[None, :],
        current_tokens,
        [current_tokens[idx]],
        f"Token '{current_tokens[idx]}' attention – Layer {layer}, Head {head}",
        cmap="magma"
    )


def compare_layers(head, token):
    if current_tokens is None or current_attentions is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    if token not in current_tokens:
        messagebox.showerror("Error", f"Token '{token}' not found.\nTokens: {current_tokens}")
        return
    idx = current_tokens.index(token)
    rows = []
    for layer in range(len(current_attentions)):
        attn = current_attentions[layer][0, head].detach().cpu().numpy()
        rows.append(attn[idx])
    layer_attn = np.stack(rows, axis=0)
    ylabels = [f"L{l}" for l in range(len(current_attentions))]
    show_heatmap(
        layer_attn,
        current_tokens,
        ylabels,
        f"Attention of '{token}' across layers (Head {head})",
        cmap="cubehelix"
    )


def compare_heads(layer, token):
    if current_tokens is None or current_attentions is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    if token not in current_tokens:
        messagebox.showerror("Error", f"Token '{token}' not found.\nTokens: {current_tokens}")
        return
    idx = current_tokens.index(token)
    rows = []
    num_heads = current_attentions[layer].shape[1]
    for head in range(num_heads):
        attn = current_attentions[layer][0, head].detach().cpu().numpy()
        rows.append(attn[idx])
    head_attn = np.stack(rows, axis=0)
    ylabels = [f"H{h}" for h in range(num_heads)]
    show_heatmap(
        head_attn,
        current_tokens,
        ylabels,
        f"Attention of '{token}' across heads (Layer {layer})",
        cmap="plasma"
    )


# -----------------------------
# GUI callbacks
# -----------------------------
def on_encode_sentence():
    sentence = sentence_entry.get().strip()
    if not sentence:
        messagebox.showerror("Error", "Please enter a sentence.")
        return
    encode_sentence(sentence)
    tokens_label.config(text="Tokens: " + "  ".join(current_tokens))
    messagebox.showinfo("Encoded", "Sentence encoded.\n[CLS] vector is ready.")


def on_show_cls_vector():
    if current_cls_embedding is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    show_vector_bar(current_cls_embedding, title="Current [CLS] sentence embedding")


def on_add_to_db():
    add_to_vector_db()
    db_count_label.config(text=f"DB entries: {len(vector_db)}")


def on_show_db_entry_vector():
    if not vector_db:
        messagebox.showerror("Error", "Vector DB is empty.")
        return
    idx_str = db_index_entry.get().strip()
    if not idx_str:
        messagebox.showerror("Error", "Enter DB index (0-based).")
        return
    try:
        idx = int(idx_str)
    except ValueError:
        messagebox.showerror("Error", "DB index must be an integer.")
        return
    if idx < 0 or idx >= len(vector_db):
        messagebox.showerror("Error", f"Index out of range. DB size: {len(vector_db)}")
        return
    text, emb = vector_db[idx]
    show_vector_bar(emb, title=f"DB[{idx}] embedding: {text}")


def on_run_query():
    query = query_entry.get().strip()
    if not query:
        messagebox.showerror("Error", "Enter a query sentence.")
        return
    if not vector_db:
        messagebox.showerror("Error", "Vector DB is empty. Add some sentences first.")
        return
    # encode query
    inputs = tokenizer(query, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    q_cls = outputs.last_hidden_state[:, 0, :][0].detach().cpu().numpy()
    # search
    results = search_vector_db(q_cls, top_k=5)
    # show results
    result_text = "Query:\n" + query + "\n\nTop matches:\n\n"
    for score, text in results:
        result_text += f"{score:.3f}  →  {text}\n"
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, result_text)
    result_box.config(state="disabled")


def on_cls_attention():
    try:
        layer = int(layer_spin.get())
        head = int(head_spin.get())
    except ValueError:
        messagebox.showerror("Error", "Layer and head must be integers.")
        return
    visualize_cls_attention(layer, head)


def on_token_attention():
    token = token_entry.get().strip()
    if not token:
        messagebox.showerror("Error", "Enter a token (exact as shown in tokens).")
        return
    try:
        layer = int(layer_spin.get())
        head = int(head_spin.get())
    except ValueError:
        messagebox.showerror("Error", "Layer and head must be integers.")
        return
    visualize_token_attention(layer, head, token)


def on_compare_layers():
    token = token_entry.get().strip()
    if not token:
        messagebox.showerror("Error", "Enter a token (exact as shown in tokens).")
        return
    try:
        head = int(head_spin.get())
    except ValueError:
        messagebox.showerror("Error", "Head must be an integer.")
        return
    compare_layers(head, token)


def on_compare_heads():
    token = token_entry.get().strip()
    if not token:
        messagebox.showerror("Error", "Enter a token (exact as shown in tokens).")
        return
    try:
        layer = int(layer_spin.get())
    except ValueError:
        messagebox.showerror("Error", "Layer must be an integer.")
        return
    compare_heads(layer, token)


# -----------------------------
# Build GUI
# -----------------------------
root = tk.Tk()
root.title("BERT + Vector DB Learning Tool")

main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0, sticky="nsew")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Sentence input
ttk.Label(main_frame, text="Sentence (to encode & store):").grid(row=0, column=0, sticky="w")
sentence_entry = ttk.Entry(main_frame, width=80)
sentence_entry.grid(row=1, column=0, columnspan=4, sticky="we", pady=5)

encode_button = ttk.Button(main_frame, text="Encode sentence", command=on_encode_sentence)
encode_button.grid(row=1, column=4, padx=5)

tokens_label = ttk.Label(main_frame, text="Tokens: (encode a sentence first)")
tokens_label.grid(row=2, column=0, columnspan=5, sticky="w", pady=5)

# CLS vector + DB controls
cls_vec_button = ttk.Button(main_frame, text="Show [CLS] vector", command=on_show_cls_vector)
cls_vec_button.grid(row=3, column=0, pady=5, sticky="we")

add_db_button = ttk.Button(main_frame, text="Add sentence to Vector DB", command=on_add_to_db)
add_db_button.grid(row=3, column=1, pady=5, sticky="we")

db_count_label = ttk.Label(main_frame, text="DB entries: 0")
db_count_label.grid(row=3, column=2, sticky="w")

ttk.Label(main_frame, text="Show DB entry index:").grid(row=3, column=3, sticky="e")
db_index_entry = ttk.Entry(main_frame, width=5)
db_index_entry.grid(row=3, column=4, sticky="w")
show_db_vec_button = ttk.Button(main_frame, text="Show DB vector", command=on_show_db_entry_vector)
show_db_vec_button.grid(row=4, column=4, sticky="we", pady=5)

# Layer / head / token controls
ttk.Label(main_frame, text="Layer (0–11):").grid(row=5, column=0, sticky="w", pady=(10, 0))
layer_spin = ttk.Spinbox(main_frame, from_=0, to=11, width=5)
layer_spin.set(0)
layer_spin.grid(row=5, column=1, sticky="w", padx=5, pady=(10, 0))

ttk.Label(main_frame, text="Head (0–11):").grid(row=5, column=2, sticky="w", pady=(10, 0))
head_spin = ttk.Spinbox(main_frame, from_=0, to=11, width=5)
head_spin.set(0)
head_spin.grid(row=5, column=3, sticky="w", padx=5, pady=(10, 0))

ttk.Label(main_frame, text="Token (exact as in tokens):").grid(row=6, column=0, sticky="w")
token_entry = ttk.Entry(main_frame, width=20)
token_entry.grid(row=6, column=1, sticky="w")

cls_att_button = ttk.Button(main_frame, text="[CLS] attention", command=on_cls_attention)
cls_att_button.grid(row=7, column=0, pady=5, sticky="we")

tok_att_button = ttk.Button(main_frame, text="Token attention", command=on_token_attention)
tok_att_button.grid(row=7, column=1, pady=5, sticky="we")

layers_button = ttk.Button(main_frame, text="Compare layers", command=on_compare_layers)
layers_button.grid(row=7, column=2, pady=5, sticky="we")

heads_button = ttk.Button(main_frame, text="Compare heads", command=on_compare_heads)
heads_button.grid(row=7, column=3, pady=5, sticky="we")

# Query + search
ttk.Label(main_frame, text="Query sentence (for similarity search):").grid(row=8, column=0, sticky="w", pady=(10, 0))
query_entry = ttk.Entry(main_frame, width=80)
query_entry.grid(row=9, column=0, columnspan=4, sticky="we", pady=5)

query_button = ttk.Button(main_frame, text="Run query against Vector DB", command=on_run_query)
query_button.grid(row=9, column=4, sticky="we", padx=5)

result_box = tk.Text(main_frame, width=90, height=8, state="disabled")
result_box.grid(row=10, column=0, columnspan=5, pady=10, sticky="we")

quit_button = ttk.Button(main_frame, text="Quit", command=root.destroy)
quit_button.grid(row=11, column=4, pady=5, sticky="e")

root.mainloop()

