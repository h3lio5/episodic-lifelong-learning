import transformers
import torch
import torch.nn as nn


class EncDec(object):
    """
    """

    def __init__(self, mode='train', model_state=None):
        super(EncDec, self).__init__()
        # Initialize the pretrained bert model for sequence classification
        if mode == 'train':
            # from_pretrained() loads weights, config from the files
            # pytorch_model.bin and config.json if available in the directory provided
            self.classifier = transformers.BertForSequenceClassification.from_pretrained(
                '../pretrained_bert_tc/model_config')
            # If weigths and config not saved locally then the model will be downloaded
            # self.classifier = transformers.BertForSequenceClassification.from_pretrained(
            #    'bert-base-uncased', num_labels=33)

        elif mode == 'test':
            # If config file not locally available, then
            # config = transformers.BertConfig.from_pretrained('bert-base-uncased', num_labels=33)
            config = transformers.BertConfig.from_pretrained(
                '../pretrained_bert_tc/model_config/config.json', num_labels=33)
            self.classifier = transformers.BertForSequenceClassification(
                config)
            self.classifier.load_state_dict(model_state['classifier'])

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
