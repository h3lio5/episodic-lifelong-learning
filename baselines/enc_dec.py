import transformers
import torch
import torch.nn as nn


class EncDec(nn.Module):
    """
    """

    def __init__(self, mode='train', model_state=None):
        super(EncDec, self).__init__()
        # Initialize the pretrained bert model for sequence classification
        if mode == 'train':
            self.classifier = transformers.BertForSequenceClassification.from_pretrained(
                '', num_labels=33)
            self.classifier.train()
        elif mode == 'test':
            # If config file not locally available, then
            # config = transformers.BertConfig.from_pretrained('bert-base-uncased', num_labels=33)
            config = transformers.BertConfig.from_pretrained(
                '../pretrained_bert/bert-base-uncased-config.json', num_labels=33)
            self.classifier = transformers.BertForSequenceClassification(
                config)
            self.classifier.load_state_dict(model_state)
            self.classifier.eval()

    def classify(self, content, attn_masks, labels):

        loss, logits = self.classifier(
            content, attention_mask=attn_masks, labels=labels)

        return loss, logits

    def infer(self, content, attn_masks):

        logits, = self.classifier(content, attention_mask=attn_masks)
        return logits

    def save_state(self):

        model_state = dict()
        model_state['classifier'] = self.classifier.state_dict()

        return model_state
