import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches


flow_selected = None
flow_canvas = None
flow_fig = None
flow_ax = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True).to(DEVICE)
model.eval()

current_sentence = None
current_tokens = None
current_attentions = None
current_cls_embedding = None

vector_db = []

def draw_flow_diagram():
    global flow_canvas, flow_fig, flow_ax

    flow_fig, flow_ax = plt.subplots(figsize=(10, 2))
    flow_ax.axis("off")

    steps = [
        "Sentence", "Tokens", "Embedding", "QKV",
        "Attention", "CLS Vector", "Vector DB",
        "Query", "Similarity", "Result"
    ]

    x_positions = [i * 1.1 for i in range(len(steps))]
    y = 0.5

    flow_ax.clear()
    flow_ax.axis("off")

    for i, (label, x) in enumerate(zip(steps, x_positions)):
        rect = patches.Rectangle((x, y), 1.0, 0.4,
                                 linewidth=2,
                                 edgecolor="gold" if flow_selected == i else "black",
                                 facecolor="white")
        flow_ax.add_patch(rect)
        flow_ax.text(x + 0.5, y + 0.2, label, ha="center", va="center")

        if i < len(steps) - 1:
            flow_ax.annotate("", xy=(x + 1.0, y + 0.2),
                             xytext=(x + 1.1, y + 0.2),
                             arrowprops=dict(arrowstyle="->"))

    flow_fig.tight_layout()

    if flow_canvas:
        flow_canvas.draw()
    else:
        flow_canvas = FigureCanvasTkAgg(flow_fig, master=diagram_frame)
        flow_canvas.get_tk_widget().pack()



# ---------------------------------------------------------
# BERT ENCODING
# ---------------------------------------------------------
def encode_sentence(sentence: str):
    global current_sentence, current_tokens, current_attentions, current_cls_embedding
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    current_attentions = outputs.attentions
    last_hidden = outputs.last_hidden_state
    current_cls_embedding = last_hidden[:, 0, :][0].detach().cpu().numpy()
    current_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    current_sentence = sentence


# ---------------------------------------------------------
# VECTOR DB
# ---------------------------------------------------------
def add_to_vector_db():
    if current_sentence is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return
    vector_db.append((current_sentence, current_cls_embedding.copy()))
    messagebox.showinfo("Vector DB", f"Added:\n{current_sentence}")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def search_vector_db(query_embedding, top_k=5):
    scores = []
    for text, emb in vector_db:
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, text))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]


# ---------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------
def show_heatmap(matrix, xlabels, ylabels, title, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=xlabels, yticklabels=ylabels, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_vector_bar(embedding, title="Vector"):
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(embedding)), embedding, color="purple")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# QKV EXPLAINER (TINY NUMBERS)
# ---------------------------------------------------------
def explain_qkv_step_by_step():
    X = np.array([1.0, 2.0, 3.0])

    Wq = np.array([
        [0.2, 0.1, 0.0],
        [0.0, 0.3, 0.1],
        [0.1, 0.0, 0.2]
    ])

    Wk = np.array([
        [0.1, 0.0, 0.2],
        [0.2, 0.1, 0.0],
        [0.0, 0.3, 0.1]
    ])

    Wv = np.array([
        [0.3, 0.1, 0.0],
        [0.0, 0.2, 0.1],
        [0.1, 0.0, 0.3]
    ])

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    score = Q.dot(K)
    attention = np.exp(score) / np.exp(score)
    output = attention * V

    text = ""
    text += "STEP 1: Input embedding X\n"
    text += f"X = {X}\n\n"

    text += "STEP 2: Weight matrices Wq, Wk, Wv\n"
    text += f"Wq =\n{Wq}\n\n"
    text += f"Wk =\n{Wk}\n\n"
    text += f"Wv =\n{Wv}\n\n"

    text += "STEP 3: Q = X × Wq\n"
    text += f"Q = {Q}\n\n"

    text += "STEP 4: K = X × Wk\n"
    text += f"K = {K}\n\n"

    text += "STEP 5: V = X × Wv\n"
    text += f"V = {V}\n\n"

    text += "STEP 6: Attention score = Q · Kᵀ\n"
    text += f"Score = {score}\n\n"

    text += "STEP 7: Softmax → attention weight\n"
    text += f"Attention = {attention}\n\n"

    text += "STEP 8: Output = Σ(attention × V)\n"
    text += f"Output = {output}\n\n"

    qkv_output.delete("1.0", tk.END)
    qkv_output.insert(tk.END, text)


# ---------------------------------------------------------
# QUERY → VECTOR → RETRIEVAL EXPLAINER
# ---------------------------------------------------------
def explain_query_flow():
    query = query_entry.get().strip()
    if not query:
        messagebox.showerror("Error", "Enter a query sentence.")
        return
    if not vector_db:
        messagebox.showerror("Error", "Vector DB is empty.")
        return

    inputs = tokenizer(query, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    q_vec = outputs.last_hidden_state[:, 0, :][0].detach().cpu().numpy()
    results = search_vector_db(q_vec)

    text = ""
    text += "STEP 1: Query sentence\n"
    text += f"{query}\n\n"

    text += "STEP 2: Query vector (first 10 dims)\n"
    text += f"{q_vec[:10]}\n\n"

    text += "STEP 3: Cosine similarity with DB entries\n\n"
    for score, sent in results:
        text += f"{score:.4f} → {sent}\n"

    best_score, best_text = results[0]
    best_emb = None
    for t, e in vector_db:
        if t == best_text:
            best_emb = e
            break

    text += "\nSTEP 4: Why this match was chosen\n"
    text += f"Best match: \"{best_text}\"\n"
    text += f"Cosine similarity: {best_score:.4f}\n"
    text += "High similarity means the angle between vectors is small.\n"

    query_output.delete("1.0", tk.END)
    query_output.insert(tk.END, text)

    if best_emb is not None:
        diff = q_vec[:20] - best_emb[:20]
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(diff)), diff, color="orange")
        plt.title("Query vs Best Match (difference in first 20 dims)")
        plt.xlabel("Dimension")
        plt.ylabel("Query - Match")
        plt.tight_layout()
        plt.show()


qkv_step = 0

def animate_qkv_step():
    global qkv_step
    X = np.array([1.0, 2.0, 3.0])

    Wq = np.array([
        [0.2, 0.1, 0.0],
        [0.0, 0.3, 0.1],
        [0.1, 0.0, 0.2]
    ])

    Wk = np.array([
        [0.1, 0.0, 0.2],
        [0.2, 0.1, 0.0],
        [0.0, 0.3, 0.1]
    ])

    Wv = np.array([
        [0.3, 0.1, 0.0],
        [0.0, 0.2, 0.1],
        [0.1, 0.0, 0.3]
    ])

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    score = Q.dot(K)
    attention = np.exp(score) / np.exp(score)
    output = attention * V

    text = ""

    if qkv_step == 0:
        text += "STEP 1: X (input embedding)\n"
        text += f"X = {X}\n"
    elif qkv_step == 1:
        text += "STEP 2: Wq, Wk, Wv\n"
        text += f"Wq =\n{Wq}\n\nWk =\n{Wk}\n\nWv =\n{Wv}\n"
    elif qkv_step == 2:
        text += "STEP 3: Q = X × Wq\n"
        text += f"Q = {Q}\n"
    elif qkv_step == 3:
        text += "STEP 4: K = X × Wk\n"
        text += f"K = {K}\n"
    elif qkv_step == 4:
        text += "STEP 5: V = X × Wv\n"
        text += f"V = {V}\n"
    elif qkv_step == 5:
        text += "STEP 6: score = Q · Kᵀ\n"
        text += f"score = {score}\n"
    elif qkv_step == 6:
        text += "STEP 7: attention = softmax(score)\n"
        text += f"attention = {attention}\n"
    elif qkv_step == 7:
        text += "STEP 8: output = attention × V\n"
        text += f"output = {output}\n"
    else:
        qkv_step = -1
        text += "Restarting QKV animation.\nNext click goes to STEP 1."

    qkv_step += 1

    qkv_output.delete("1.0", tk.END)
    qkv_output.insert(tk.END, text)


def show_sentence_step():
    text = "STEP: Sentence\n\n"
    text += f"Original sentence:\n{current_sentence}\n\n"
    text += "This is the raw input before any processing."
    qkv_output.delete("1.0", tk.END)
    qkv_output.insert(tk.END, text)


def show_tokens_step():
    if current_tokens is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return

    ids = tokenizer.convert_tokens_to_ids(current_tokens)

    text = "STEP: Tokenization\n\n"
    text += "Tokens:\n" + "  ".join(current_tokens) + "\n\n"
    text += "Token IDs:\n" + "  ".join(str(i) for i in ids) + "\n\n"
    text += "Meaning: BERT splits words into subwords and assigns each an ID."

    qkv_output.delete("1.0", tk.END)
    qkv_output.insert(tk.END, text)


def show_embedding_step():
    if current_tokens is None:
        messagebox.showerror("Error", "Encode a sentence first.")
        return

    inputs = tokenizer(current_sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    emb = outputs.hidden_states[0][0].cpu().numpy()  # input embeddings

    text = "STEP: Embedding (X)\n\n"
    text += "Each token is converted into a 768‑dimensional vector.\n"
    text += "Below are the first 16 dimensions for each token:\n\n"

    for tok, vec in zip(current_tokens, emb):
        text += f"{tok:10s} → {vec[:16]}\n"

    qkv_output.delete("1.0", tk.END)
    qkv_output.insert(tk.END, text)


def show_qkv_step():
    text = "STEP: Q / K / V Transformation\n\n"
    text += "BERT computes three vectors for each token:\n"
    text += "• Q = Query (what am I looking for?)\n"
    text += "• K = Key (what do I contain?)\n"
    text += "• V = Value (what information do I carry?)\n\n"

    text += "Real BERT shapes:\n"
    text += "X: (seq_len, 768)\n"
    text += "Wq, Wk, Wv: (768 × 768)\n"
    text += "Q, K, V: (seq_len, 768)\n\n"

    text += "Tiny example (3‑dim) for teaching:\n"
    text += "X = [1, 2, 3]\n"
    text += "Wq, Wk, Wv = 3×3 matrices\n"
    text += "Q = X·Wq, K = X·Wk, V = X·Wv\n\n"
    text += "Click 'Next QKV Step' to animate the math."

    qkv_output.delete("1.0", tk.END)
    qkv_output.insert(tk.END, text)


def show_vector_db_map():
    if not vector_db:
        messagebox.showerror("Error", "Vector DB is empty.")
        return

    embeddings = np.array([emb for _, emb in vector_db])
    texts = [txt for txt, _ in vector_db]

    if embeddings.shape[0] < 2:
        messagebox.showerror("Error", "Need at least 2 entries for a map.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c="blue")

    for (x, y), label in zip(coords, texts):
        plt.text(x, y, label[:15] + ("..." if len(label) > 15 else ""), fontsize=8)

    plt.title("Vector DB 2D Map (PCA)")
    plt.tight_layout()
    plt.show()

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

flow_selected = None
flow_canvas = None
flow_fig = None
flow_ax = None

def draw_flow_diagram():
    global flow_canvas, flow_fig, flow_ax

    steps = [
        "Sentence", "Tokens", "Embedding", "QKV",
        "Attention", "CLS", "Vector DB",
        "Query", "Similarity", "Result"
    ]

    x_positions = [i * 1.2 for i in range(len(steps))]
    y = 0.5

    flow_fig, flow_ax = plt.subplots(figsize=(12, 2))
    flow_ax.axis("off")

    for i, (label, x) in enumerate(zip(steps, x_positions)):
        rect = patches.Rectangle(
            (x, y), 1.0, 0.4,
            linewidth=3,
            edgecolor="gold" if flow_selected == i else "black",
            facecolor="white"
        )
        flow_ax.add_patch(rect)
        flow_ax.text(x + 0.5, y + 0.2, label, ha="center", va="center")

        if i < len(steps) - 1:
            flow_ax.annotate(
                "",
                xy=(x + 1.0, y + 0.2),
                xytext=(x + 1.2, y + 0.2),
                arrowprops=dict(arrowstyle="->", lw=2)
            )

    flow_fig.tight_layout()

    if flow_canvas:
        flow_canvas.draw()
    else:
        flow_canvas = FigureCanvasTkAgg(flow_fig, master=diagram_frame)
        flow_canvas.get_tk_widget().pack()


def draw_flow_diagram():
    global flow_canvas, flow_fig, flow_ax

    flow_fig, flow_ax = plt.subplots(figsize=(10, 2))
    flow_ax.axis("off")

    steps = [
        "Sentence", "Tokens", "Embedding", "QKV",
        "Attention", "CLS Vector", "Vector DB",
        "Query", "Similarity", "Result"
    ]

    x_positions = [i * 1.1 for i in range(len(steps))]
    y = 0.5

    flow_ax.clear()
    flow_ax.axis("off")

    for i, (label, x) in enumerate(zip(steps, x_positions)):
        rect = patches.Rectangle((x, y), 1.0, 0.4,
                                 linewidth=2,
                                 edgecolor="gold" if flow_selected == i else "black",
                                 facecolor="white")
        flow_ax.add_patch(rect)
        flow_ax.text(x + 0.5, y + 0.2, label, ha="center", va="center")

        if i < len(steps) - 1:
            flow_ax.annotate("", xy=(x + 1.0, y + 0.2),
                             xytext=(x + 1.1, y + 0.2),
                             arrowprops=dict(arrowstyle="->"))

    flow_fig.tight_layout()

    if flow_canvas:
        flow_canvas.draw()
    else:
        flow_canvas = FigureCanvasTkAgg(flow_fig, master=diagram_frame)
        flow_canvas.get_tk_widget().pack()



def select_flow_step(step_index):
    global flow_selected
    flow_selected = step_index
    draw_flow_diagram()

    if step_index == 0: show_sentence_step()
    elif step_index == 1: show_tokens_step()
    elif step_index == 2: show_embedding_step()
    elif step_index == 3: show_qkv_step()
    elif step_index == 4: on_cls_attention()
    elif step_index == 5: on_show_cls_vector()
    elif step_index == 6: on_add_to_db()
    elif step_index == 7: explain_query_flow()
    elif step_index == 8: explain_query_flow()
    elif step_index == 9: explain_query_flow()



def next_pipeline_step():
    global pipeline_step
    select_flow_step(pipeline_step)

    if pipeline_step < 9:
        animate_arrow(pipeline_step, pipeline_step + 1)

    pipeline_step += 1
    if pipeline_step > 9:
        pipeline_step = 0


def lesson_mode_step():
    next_pipeline_step()
    root.after(1500, lesson_mode_step)
    animate_arrow(0, 1)  # Sentence → Tokens



def animate_arrow(step_from, step_to):
    x1 = step_from * 1.1 + 1.0
    x2 = step_to * 1.1
    y = 0.7

    flow_ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", lw=3, color="red"))
    flow_canvas.draw()


qkv_step = 0

# ---------------------------------------------------------
# GUI
# ---------------------------------------------------------
root = tk.Tk()
root.title("BERT + Vector DB + QKV Explainer")

main = ttk.Frame(root, padding=10)
diagram_frame = ttk.LabelFrame(main, text="Flow Diagram")
diagram_frame.grid(row=9, column=0, columnspan=4, pady=10, sticky="we")
draw_flow_diagram(diagram_frame)

main.grid(row=0, column=0)

# Sentence input
ttk.Label(main, text="Sentence to encode:").grid(row=0, column=0, sticky="w")
sentence_entry = ttk.Entry(main, width=80)
sentence_entry.grid(row=1, column=0, columnspan=3)

ttk.Button(main, text="Encode Sentence", command=lambda: (
    encode_sentence(sentence_entry.get()),
    tokens_label.config(text="Tokens: " + " ".join(current_tokens))
)).grid(row=1, column=3)

tokens_label = ttk.Label(main, text="Tokens: (none)")
tokens_label.grid(row=2, column=0, columnspan=4, sticky="w")

# Vector DB
ttk.Button(main, text="Add to Vector DB", command=add_to_vector_db).grid(row=3, column=0)
ttk.Button(main, text="Show CLS Vector", command=lambda: show_vector_bar(current_cls_embedding)).grid(row=3, column=1)
ttk.Button(main, text="Show Vector DB Map", command=show_vector_db_map).grid(row=3, column=2)
ttk.Button(main, text="Next Pipeline Step", command=next_pipeline_step).grid(row=13, column=0, pady=10)
ttk.Button(main, text="Start Lesson Mode", command=lesson_mode_step).grid(row=13, column=1, pady=10)




# QKV Explainer
ttk.Label(main, text="QKV Step-by-Step Explanation:").grid(row=4, column=0, sticky="w", pady=10)
qkv_output = tk.Text(main, width=100, height=15)
qkv_output.grid(row=5, column=0, columnspan=4)
ttk.Button(main, text="Run QKV Explanation", command=explain_qkv_step_by_step).grid(row=4, column=3)
ttk.Button(main, text="Next QKV Step", command=animate_qkv_step).grid(row=4, column=2)


# Query flow
ttk.Label(main, text="Query sentence:").grid(row=6, column=0, sticky="w", pady=10)
query_entry = ttk.Entry(main, width=80)
query_entry.grid(row=7, column=0, columnspan=3)

ttk.Button(main, text="Explain Query → Vector → Retrieval", command=explain_query_flow).grid(row=7, column=3)

query_output = tk.Text(main, width=100, height=15)
query_output.grid(row=8, column=0, columnspan=4)

diagram_frame = ttk.LabelFrame(main, text="Flow Diagram (Click a Step)")
diagram_frame.grid(row=12, column=0, columnspan=4, pady=10, sticky="we")

# clickable buttons
for i, label in enumerate(["Sentence", "Tokens", "Embedding", "QKV",
                           "Attention", "CLS", "DB", "Query", "Similarity", "Result"]):
    ttk.Button(diagram_frame, text=label,
               command=lambda idx=i: select_flow_step(idx)).grid(row=0, column=i)


draw_flow_diagram()


# New Frame
diagram_frame = ttk.LabelFrame(main, text="Flow Diagram (Click any step)")
diagram_frame.grid(row=12, column=0, columnspan=4, pady=10, sticky="we")
btn_sentence = ttk.Button(diagram_frame, text="Sentence", command=show_sentence_step)
btn_sentence.grid(row=0, column=0, padx=5, pady=5)

btn_tokens = ttk.Button(diagram_frame, text="Tokens", command=show_tokens_step)
btn_tokens.grid(row=0, column=1, padx=5, pady=5)

btn_embedding = ttk.Button(diagram_frame, text="Embedding (X)", command=show_embedding_step)
btn_embedding.grid(row=0, column=2, padx=5, pady=5)

btn_qkv = ttk.Button(diagram_frame, text="Q / K / V", command=show_qkv_step)
btn_qkv.grid(row=0, column=3, padx=5, pady=5)




root.mainloop()

