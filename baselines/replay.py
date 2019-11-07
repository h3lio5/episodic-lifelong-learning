import torch
import random
import transformers


class Replay(object):
    """
    Stores the examples for sparse experience replay
    """

    def __init__(self):
        """
        Initialize the replay buffer
        """
        self.memory = []
        # Initialize the pretrained bert model for sequence classification
        self.classifier = transformers.BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=33)

    def classify(self, content, attn_masks, labels):

        self.classifier.train()
        loss, logits = self.classifier(
            content, attention_mask=attn_masks, labels=labels)
        return loss, logits

    def infer(self, content, attn_masks):

        self.classifier.eval()
        logits = self.classifier(content, attention_mask=attn_masks)
        return logits

    def push(self, examples):
        """
        Add the examples as tuples of individual content,attention_mask,label to the replay buffer
        """

        _batch = []

        for content, attn_mask, label in zip(examples):
            _batch.append(
                (content.cpu().numpy(), attn_mask.cpu().numpy(), label.cpu().numpy()))

        self.memory.extend(batch)

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

        for content, attn_mask, label in samples:
            contents.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)

        return (torch.LongTensor([contents]), torch.LongTensor([attn_masks]), torch.LongTensor([labels]))
