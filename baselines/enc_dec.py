import transformers
import torch
import torch.nn as nn


class EncDec(nn.Module):
    """
    """

    def __init__(self):
        super(EncDec,self).__init__()
        # Initialize the pretrained bert model for sequence classification
        self.classifier = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=33)


    def classify(self,content,attn_masks,labels):


        loss,logits = self.classifier(content,attention_mask=attn_masks,labels=labels)

        return loss,logits

    def infer(self,content,attn_masks):


        logits = self.classifier(content,attention_mask=attn_masks)
        return logits