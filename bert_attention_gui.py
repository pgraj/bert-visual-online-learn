import tkinter as tk
from tkinter import ttk, messagebox

import torch
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

current_tokens = None
current_attentions = None


# -----------------------------
# Core BERT + attention logic
# -----------------------------
def run_bert(sentence: str):
    global current_tokens, current_attentions
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    current_attentions = outputs.attentions
    current_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])


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


def visualize_cls_attention(layer, head):
    if current_tokens is None or current_attentions is None:
        messagebox.showerror("Error", "Run BERT on a sentence first.")
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
        messagebox.showerror("Error", "Run BERT on a sentence first.")
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
        messagebox.showerror("Error", "Run BERT on a sentence first.")
        return
    if token not in current_tokens:
        messagebox.showerror("Error", f"Token '{token}' not found.\nTokens: {current_tokens}")
        return
    idx = current_tokens.index(token)
    rows = []
    for layer in range(len(current_attentions)):
        attn = current_attentions[layer][0, head].detach().cpu().numpy()
        rows.append(attn[idx])
    import numpy as np
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
        messagebox.showerror("Error", "Run BERT on a sentence first.")
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
    import numpy as np
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
def on_run_bert():
    sentence = sentence_entry.get().strip()
    if not sentence:
        messagebox.showerror("Error", "Please enter a sentence.")
        return
    run_bert(sentence)
    tokens_label.config(text="Tokens: " + "  ".join(current_tokens))


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
root.title("BERT Attention Learning Tool")

main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0, sticky="nsew")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Sentence input
ttk.Label(main_frame, text="Enter a sentence:").grid(row=0, column=0, sticky="w")
sentence_entry = ttk.Entry(main_frame, width=80)
sentence_entry.grid(row=1, column=0, columnspan=4, sticky="we", pady=5)

run_button = ttk.Button(main_frame, text="Run BERT", command=on_run_bert)
run_button.grid(row=1, column=4, padx=5)

# Tokens display
tokens_label = ttk.Label(main_frame, text="Tokens: (run BERT first)")
tokens_label.grid(row=2, column=0, columnspan=5, sticky="w", pady=5)

# Layer / head controls
ttk.Label(main_frame, text="Layer (0–11):").grid(row=3, column=0, sticky="w")
layer_spin = ttk.Spinbox(main_frame, from_=0, to=11, width=5)
layer_spin.set(0)
layer_spin.grid(row=3, column=1, sticky="w", padx=5)

ttk.Label(main_frame, text="Head (0–11):").grid(row=3, column=2, sticky="w")
head_spin = ttk.Spinbox(main_frame, from_=0, to=11, width=5)
head_spin.set(0)
head_spin.grid(row=3, column=3, sticky="w", padx=5)

# Token entry
ttk.Label(main_frame, text="Token (exact as in tokens):").grid(row=4, column=0, sticky="w", pady=(10, 0))
token_entry = ttk.Entry(main_frame, width=20)
token_entry.grid(row=4, column=1, sticky="w", pady=(10, 0))

# Buttons for visualizations
cls_button = ttk.Button(main_frame, text="[CLS] attention", command=on_cls_attention)
cls_button.grid(row=5, column=0, pady=10, sticky="we")

token_button = ttk.Button(main_frame, text="Token attention", command=on_token_attention)
token_button.grid(row=5, column=1, pady=10, sticky="we")

layers_button = ttk.Button(main_frame, text="Compare layers", command=on_compare_layers)
layers_button.grid(row=5, column=2, pady=10, sticky="we")

heads_button = ttk.Button(main_frame, text="Compare heads", command=on_compare_heads)
heads_button.grid(row=5, column=3, pady=10, sticky="we")

quit_button = ttk.Button(main_frame, text="Quit", command=root.destroy)
quit_button.grid(row=5, column=4, pady=10, sticky="we")

root.mainloop()
