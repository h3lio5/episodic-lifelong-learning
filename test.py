import transformers

print("started ")
model = transformers.BertModel.from_pretrained(
    'bert-base-uncased')
print("saving")
model.save_pretrained('../pretrained_bert_tc/key_encoder')
print("saved")
