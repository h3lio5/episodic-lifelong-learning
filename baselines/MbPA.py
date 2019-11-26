import torch
import torch.nn as nn
import transformers
import numpy as np
import random


class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self):

        self.memory = {}

    def push(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        contents, attn_masks, labels = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            self.memory.update({key: (contents[i], attn_masks[i], labels[i])})


class MbPA():
    """
    Implements Memory based Parameter Adaptation model
    """

    def __init__(self, K=32, L=30, retrieval='nearest', use_cuda=True):
        super(MbPA).__init__(self)
        # Key network to find key representation of content
        self.key_enc = transformers.BertModel.from_pretrained(
            'bert-base-uncased')
        # Bert model for text classification
        self.classifier = transformers.BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=33)
        # the base model which needs to be locally adapted
        self.model = model
        # Memory buffer storing episodes(examples)
        self.memory = {}
        # number of neighbours to retrieve from the memory buffer
        self.K = K
        # Number of local adaptation steps
        self.L = L
        # mode of retrieval of examples from memory
        self.mode = retrieval
        self.use_cuda = use_cuda

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
            last_hidden_states, _ = self.key_enc(
                contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        keys = last_hidden_states[:, 0, :]

        return keys

    def infer(self, content_batch):
        """
        Infer the labels from classifier after locally adapting the parameters based on memory
        """
        # weights of the base model
        base_weights = model.parameters()
        for param in base_weights:
            param.requires_grad = False
        # store the output labels
        out_labels = []

        for content in content_batch:
            # new model that is to be locally adapted based on the content
            new_model = type(model)()
            new_model.load_state_dict(model.state_dict())
            # Freeze the key network weights
            with torch.no_grad():
                keys, _ = self.key_enc(content.unsqueeze(0))
            key = keys[0].numpy()
            # Get k nearest/random neighbours
            neighbour_keys = self._get_neighbour_keys(key)
            k_content = []
            k_attn_masks = []
            k_labels = []

            for key in neighbour_keys:
                sample = self.memory[key]
                k_content.append(sample[0])
                k_attn_masks.append(sample[1])
                k_labels.append(sample[2])

            # Convert the lists into tensors
            k_content = torch.LongTensor(k_content)
            k_attn_masks = torch.LongTensor(k_attn_masks)
            k_labels = torch.LongTensor(k_labels)

            if self.use_cuda:
                new_model = new_model.cuda()
                k_content = k_content.cuda()
                k_attn_masks = k_attn_masks.cuda()
                k_labels = k_labels.cuda()

            optimizer = transformers.AdamW(new_model.parameters(), lr=1e-3)
            # Train the base model for L epochs to locally adapt the parameters : modify the
            # base network weights to fit the k retrieved data samples from the memory
            for epoch in range(self.L):
                # zero out the gradients
                optimizer.zero_grad()
                likelihood_loss, logits = new_model(
                    k_content, attention_mask=k_attn_masks, labels=k_labels)
                # Current model weights
                curr_weights = new_model.parameters()
                diff_loss = 0
                # Iterate over base_weights and curr_weights and accumulate the euclidean norm
                # of their differences
                for base_param, curr_param in zip(base_weights, curr_weights):
                    diff = (base_param-curr_param).pow(2).sum()
                    diff_loss += diff
                t_loss = likelihood_loss + 0.001*diff_loss
                t_loss.backward()
                optimizer.step()

            logits = new_model(content.unsqueeze(0))
            ans_label = torch.argmax(logits)
            out_labels.append(ans_label.numpy())

        return np.asarray(out_labels)

    def _get_neighbour_keys(self, key):
        """
        Retrives k examples from the episodic memory using nearest neighbour approach or randomly
        """
        self.K = k
        if self.mode == 'nearest':
            all_keys = np.asarray(list(self.memory.keys()))
            similarity_scores = np.dot(all_keys, key.T)
            K_neighbour_keys = all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            return K_neighbour_keys

        elif self.mode == 'random':
            all_keys = np.asarray(list(self.memory.keys()))
            k_random_keys = random.sample(all_keys, k)
            return k_random_keys

    def push(self, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        contents, attn_masks, labels = examples
        # Freeze the weights of the key network
        with torch.no_grad():
            last_hidden_states, _ = self.key_enc(
                contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        keys = last_hidden_states[:, 0, :].cpu().numpy()
        # update the memory dictionary
        for i, key in enumerate(keys):
            self.memory.update({key: (contents[i].cpu().numpy(
            ), attn_masks[i].cpu().numpy(), labels[i].cpu().numpy())})
