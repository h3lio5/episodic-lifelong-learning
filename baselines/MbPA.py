import torch
import torch.nn as nn
import transformers
import numpy as np
from tqdm import trange
import copy
import pdb


class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, buffer=None):

        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer
            total_keys = len(buffer.keys())
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, 768)

    def push(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        contents, attn_masks, labels = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key
            self.memory.update(
                {key.tobytes(): (contents[i], attn_masks[i], labels[i])})

    def _prepare_batch(self, sample):
        """
        Parameter:
        sample -> list of tuple of experiences
               -> i.e, [(content_1,attn_mask_1,label_1),.....,(content_k,attn_mask_k,label_k)]
        Returns:
        batch -> tuple of list of content,attn_mask,label
              -> i.e, ([content_1,...,content_k],[attn_mask_1,...,attn_mask_k],[label_1,...,label_k])
        """
        contents = []
        attn_masks = []
        labels = []
        # Iterate over experiences
        for content, attn_mask, label in sample:
            # convert the batch elements into torch.LongTensor
            contents.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)

        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels))

    def get_neighbours(self, keys, k=32):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
            # converts experiences into batch
            batch = self._prepare_batch(neighbours)
            samples.append(batch)

        return samples


class MbPA(nn.Module):
    """
    Implements Memory based Parameter Adaptation model
    """

    def __init__(self, L=30, model_state=None):
        super(MbPA, self).__init__()

        if model_state is None:
            # Key network to find key representation of content
            self.key_encoder = transformers.BertModel.from_pretrained(
                '../pretrained_bert_tc/key_encoder')
            # Bert model for text classification
            self.classifier = transformers.BertForSequenceClassification.from_pretrained(
                '../pretrained_bert_tc/classifier')

        else:

            cls_config = transformers.BertConfig.from_pretrained(
                '../pretrained_bert_tc/classifier/config.json', num_labels=33)
            self.classifier = transformers.BertForSequenceClassification(
                cls_config)
            self.classifier.load_state_dict(model_state['classifier'])
            key_config = transformers.BertConfig.from_pretrained(
                '../pretrained_bert_tc/key_encoder/config.json')
            self.key_encoder = transformers.BertModel(key_config)
            self.key_encoder.load_state_dict(model_state['key_encoder'])
            # base model weights
            self.base_weights = list()
            # # Freeze the base model weights
            for param in self.classifier.parameters():
                self.base_weights.append(param.data.cuda())

        # Number of local adaptation steps
        self.L = L

    def classify(self, content, attention_mask, labels):
        """
        Bert classification model
        """
        loss, logits = self.classifier(
            content, attention_mask=attention_mask, labels=labels)
        return loss, logits

    def get_keys(self, contents, attn_masks):
        """
        Return key representation of the documents
        """
        # Freeze the weights of the key network to prevent key
        # representations from drifting as data distribution changes
        with torch.no_grad():
            last_hidden_states, _ = self.key_encoder(
                contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        keys = last_hidden_states[:, 0, :]

        return keys

    def infer(self, content, attn_mask, K_contents, K_attn_masks, K_labels):
        """
        Function that performs inference based on memory based local adaptation
        Parameters:
        content   -> document that needs to be classified
        attn_mask -> attention mask over document
        rt_batch  -> the batch of samples retrieved from the memory using nearest neighbour approach

        Returns:
        logit -> label corresponding to the single document provided,i.e, content
        """

        # create a local copy of the classifier network
        adaptive_classifier = copy.deepcopy(self.classifier)
        optimizer = transformers.AdamW(
            adaptive_classifier.parameters(), lr=1e-3)

        # Current model weights
        curr_weights = list(adaptive_classifier.parameters())
        # Train the adaptive classifier for L epochs with the rt_batch
        for _ in trange(self.L, desc='Local Adaptation'):

            # zero out the gradients
            optimizer.zero_grad()
            likelihood_loss, _ = adaptive_classifier(
                K_contents, attention_mask=K_attn_masks, labels=K_labels)

            diff = torch.Tensor([0]).cuda()
            # Iterate over base_weights and curr_weights and accumulate the euclidean norm
            # of their differences
            for base_param, curr_param in zip(self.base_weights, curr_weights):
                diff += (curr_param-base_param).pow(2).sum()

            # Total loss due to log likelihood and weight restraint
            diff_loss = 0.001*diff
            diff_loss.backward()
            likelihood_loss.backward()
            optimizer.step()
        # Delete the k neigbours after training to freeup memory
        del diff_loss
        del likelihood_loss
        del K_contents
        del K_attn_masks
        del K_labels

        logits, = adaptive_classifier(content.unsqueeze(
            0), attention_mask=attn_mask.unsqueeze(0))
        del curr_weights
        del adaptive_classifier
        del optimizer
        del content
        del attn_mask
        pdb.set_trace()

        return logits

    def save_state(self):
        """
        Returns model state
        """
        model_state = dict()
        model_state['classifier'] = self.classifier.state_dict()
        model_state['key_encoder'] = self.key_encoder.state_dict()

        return model_state
