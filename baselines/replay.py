import torch
import random
import transformers
import torch.nn as nn


class ReplayMemory(object):
    """
    Stores the examples for sparse experience replay
    """

    def __init__(self):
        """
        Create the empty memory buffer
        """

        self.memory = []

    def push(self, examples):
        """
        Add the examples as tuples of individual content,attention_mask,label to the replay buffer
        """

        _batch = []
        contents, attn_masks, labels = examples
        for content, attn_mask, label in zip(contents.squeeze(1), attn_masks.squeeze(1), labels.squeeze(1)):
            _batch.append(
                (content.numpy(), attn_mask.numpy(), label.numpy()))

        self.memory.extend(_batch)

    def sample(self, sample_size=100):
        """
        Parameter:
        S : number of examples to sample from replay buffer

        Returns:
        tuple of S number of text content and their corresponding attention_masks and labels
        """
        contents = []
        attn_masks = []
        labels = []

        samples = random.sample(self.memory, sample_size)
        print("samples ", (len(samples)))
        for content, attn_mask, label in samples:
            contents.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)
        print("content ", len(contents), " type ", type(contents))
        print(contents)
        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels))


class ReplayModel(nn.Module):
    """
    Stores the examples for sparse experience replay
    """

    def __init__(self, mode='train', model_state=None):
        """
        Initialize the replay model
        """
        super(ReplayModel, self).__init__()
        # Initialize the pretrained bert model for sequence classification
        if mode == 'train':
            # from_pretrained() loads weights, config from the files
            # pytorch_model.bin and config.json if available in the directory provided
            self.classifier = transformers.BertForSequenceClassification.from_pretrained(
                '../pretrained_bert_tc/model_config')
            # If weigths and config not saved locally then initialize the model using -
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

        self.classifier.train()
        loss, logits = self.classifier(
            content, attention_mask=attn_masks, labels=labels)
        return loss, logits

    def infer(self, content, attn_masks):

        self.classifier.eval()
        logits = self.classifier(content, attention_mask=attn_masks)
        return logits

    def save_state(self):

        model_state = dict()
        model_state['classifier'] = self.classifier.state_dict()

        return model_state
