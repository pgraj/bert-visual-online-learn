# 1. Imports
from transformers import BertTokenizer, BertModel
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. Load model with attention output enabled
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# 4. Tokenize your sentence
sentence = input("Enter a sentence: ")
inputs = tokenizer(sentence, return_tensors="pt")

# 5. Run the model
outputs = model(**inputs)

# 6. Extract attention
attentions = outputs.attentions   # list of 12 layers

# 7. Convert IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 8. Pick a layer and head to visualize
layer = 0
head = 0
attn = attentions[layer][0, head].detach().numpy()

# 9. Visualize heatmap
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.show()

