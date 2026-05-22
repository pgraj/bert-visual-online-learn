import sys
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel

# -----------------------------
# Global setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
).to(DEVICE)
model.eval()


# -----------------------------
# Helper: run BERT on a sentence
# -----------------------------
def run_bert(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions  # list of (batch, heads, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, attentions


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


def visualize_cls_attention(tokens, attentions, layer=0, head=0):
    attn = attentions[layer][0, head].detach().cpu().numpy()  # (seq, seq)
    cls_row = attn[0]  # [CLS] is index 0
    show_heatmap(
        cls_row[None, :],
        tokens,
        ["[CLS]"],
        f"[CLS] attention – Layer {layer}, Head {head}",
        cmap="magma"
    )


def visualize_token_attention(tokens, attentions, layer=0, head=0, token="killed"):
    if token not in tokens:
        print(f"Token '{token}' not found in tokens: {tokens}")
        return
    idx = tokens.index(token)
    attn = attentions[layer][0, head].detach().cpu().numpy()
    row = attn[idx]
    show_heatmap(
        row[None, :],
        tokens,
        [tokens[idx]],
        f"Token '{tokens[idx]}' attention – Layer {layer}, Head {head}",
        cmap="magma"
    )


def compare_layers(tokens, attentions, head=0, token="killed"):
    if token not in tokens:
        print(f"Token '{token}' not found in tokens: {tokens}")
        return
    idx = tokens.index(token)
    rows = []
    for layer in range(len(attentions)):
        attn = attentions[layer][0, head].detach().cpu().numpy()
        rows.append(attn[idx])
    import numpy as np
    layer_attn = np.stack(rows, axis=0)  # (layers, seq)
    ylabels = [f"L{l}" for l in range(len(attentions))]
    show_heatmap(
        layer_attn,
        tokens,
        ylabels,
        f"Attention of '{token}' across layers (Head {head})",
        cmap="cubehelix"
    )


def compare_heads(tokens, attentions, layer=0, token="killed"):
    if token not in tokens:
        print(f"Token '{token}' not found in tokens: {tokens}")
        return
    idx = tokens.index(token)
    rows = []
    num_heads = attentions[layer].shape[1]
    for head in range(num_heads):
        attn = attentions[layer][0, head].detach().cpu().numpy()
        rows.append(attn[idx])
    import numpy as np
    head_attn = np.stack(rows, axis=0)  # (heads, seq)
    ylabels = [f"H{h}" for h in range(num_heads)]
    show_heatmap(
        head_attn,
        tokens,
        ylabels,
        f"Attention of '{token}' across heads (Layer {layer})",
        cmap="plasma"
    )


# -----------------------------
# Menu / main loop
# -----------------------------
def main():
    print("=" * 70)
    print("BERT Attention Learning Tool")
    print("Enter any sentence and explore BERT’s attention visually.")
    print("=" * 70)

    while True:
        sentence = input("\nEnter a sentence (or 'q' to quit): ").strip()
        if sentence.lower() in ["q", "quit", "exit"]:
            print("Exiting.")
            break

        tokens, attentions = run_bert(sentence)
        print("\nTokens:", tokens)
        print(f"Number of layers: {len(attentions)}, heads per layer: {attentions[0].shape[1]}")

        while True:
            print("\nChoose an option:")
            print("  1) [CLS] attention row")
            print("  2) Token-specific attention")
            print("  3) Compare layers for a token")
            print("  4) Compare heads for a token")
            print("  5) Enter a new sentence")
            print("  q) Quit")

            choice = input("Your choice: ").strip().lower()

            if choice == "1":
                try:
                    layer = int(input("Layer index (0–11): "))
                    head = int(input("Head index (0–11): "))
                    visualize_cls_attention(tokens, attentions, layer, head)
                except Exception as e:
                    print("Error:", e)

            elif choice == "2":
                token = input("Token to inspect (exact as shown in tokens list): ").strip()
                try:
                    layer = int(input("Layer index (0–11): "))
                    head = int(input("Head index (0–11): "))
                    visualize_token_attention(tokens, attentions, layer, head, token)
                except Exception as e:
                    print("Error:", e)

            elif choice == "3":
                token = input("Token to inspect across layers: ").strip()
                try:
                    head = int(input("Head index (0–11): "))
                    compare_layers(tokens, attentions, head, token)
                except Exception as e:
                    print("Error:", e)

            elif choice == "4":
                token = input("Token to inspect across heads: ").strip()
                try:
                    layer = int(input("Layer index (0–11): "))
                    compare_heads(tokens, attentions, layer, token)
                except Exception as e:
                    print("Error:", e)

            elif choice == "5":
                break

            elif choice in ["q", "quit", "exit"]:
                print("Exiting.")
                sys.exit(0)

            else:
                print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()

