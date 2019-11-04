import transformers 
import torch

model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=33)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
logits, = model(input_ids)
print(logits)
print(type(logits))
print(logits[0])
print(len(logits))
