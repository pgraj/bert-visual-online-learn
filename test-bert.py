from transformers import BertTokenizer, BertModel
import torch

sentence = "Rama killed Ravana"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# Real CLS embedding
cls_embedding = outputs.last_hidden_state[:, 0, :]

# Real attention maps (12 layers × 12 heads)
attentions = outputs.attentions

print(cls_embedding.shape)      # torch.Size([1, 768])
print(len(attentions))          # 12 layers
print(attentions[0].shape)      # (1, 12 heads, seq_len, seq_len)
