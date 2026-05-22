import matplotlib.pyplot as plt
import seaborn as sns

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

layer = 0      # e.g., first layer
head = 0       # e.g., first head

attn = attentions[layer][0, head].detach().numpy()  # (seq_len, seq_len)

plt.figure(figsize=(6, 5))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)
plt.title(f"Layer {layer}, Head {head}")
plt.xlabel("Key (attended to)")
plt.ylabel("Query (attending)")
plt.show()

