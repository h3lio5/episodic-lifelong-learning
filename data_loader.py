import torch
import torch.utils.data as data
import pandas as pd
from transformers import BertTokenizer


class DataSet(data.Dataset):

    def __init__(self, order=1, split='train'):

        df = pd.read_csv('data/ordered_data/'+split+'/'+str(order)+'.csv')
        self.labels = df.labels.values
        self.content = df.content.values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):
        content = self.content[index]
        content_ids = self.tokenizer.encode(content)
        padded_content_ids = self._add_spl_ids_and_pad(content_ids)
        # Create attention mask
        # Create a mask of 1s for each token followed by 0s for padding
        attention_mask = [int(i > 0) for i in padded_content_ids]
        label = self.labels[index]

        return (torch.LongTensor([padded_content_ids]), torch.LongTensor([attention_mask]), torch.LongTensor([label]))

    def _add_spl_ids_and_pad(self, input_ids, maxlen=128):

        if len(input_ids) > maxlen-2:
            input_ids = [self.tokenizer.cls_token_id] + \
                input_ids[:maxlen-2] + [self.tokenizer.sep_token_id]
            return input_ids

        output = [self.tokenizer.cls_token_id]
        output.extend(input_ids)
        output.append(self.tokenizer.sep_token_id)
        padding = [0]*(maxlen-len(output))
        output.extend(padding)

        return output
