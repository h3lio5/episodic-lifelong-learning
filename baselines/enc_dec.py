import transformers
import torch
import torch.nn as nn


class EncDec(nn.Module):
    """
    """

    def __init__(self):

        # Initialize the pretrained bert model for sequence classification
        self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=33)

    def forward(content,attn_masks,labels):

        outputs = self.model(content,attention_mask=attn_masks,labels=labels)

        return outputs[:2]
