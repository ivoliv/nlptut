import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nlpClasses import Vocabulary


class CBOWVectorizer(object):
    def __init__(self, context_vocab, target_vocab):
        self.context_vocab = context_vocab
        self.target_vocab = target_vocab
        print('context vocab:', len(self.context_vocab), 'unique tokens')
        print('target vocab: ', len(self.target_vocab), 'unique tokens')

    def vectorize_context(self, item):
        return [self.context_vocab.lookup_token(tok) for tok in eval(item.context)]

    def vectorize_target(self, item):
        return self.target_vocab.lookup_token(item.target)

    @classmethod
    def from_dataframe(cls, cbow_df):
        context_vocab = Vocabulary()
        target_vocab = Vocabulary(add_unk=False)

        for index, row in cbow_df.iterrows():
            for c in eval(row.context):
                context_vocab.add_token(c)
            target_vocab.add_token(row.target)

        return cls(context_vocab, target_vocab)


class CBOWDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('torch device: ', self.device)
        self.text_df = text_df
        self.vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df['split'] == 'train'].reset_index()
        self.train_size = len(self.train_df)

        self.valid_df = self.text_df[self.text_df['split'] == 'valid'].reset_index()
        self.valid_size = len(self.valid_df)

        self.test_df = self.text_df[self.text_df['split'] == 'test'].reset_index()
        self.test_size = len(self.test_df)

        self.lookup_dict = {'train': (self.train_df, self.train_size),
                            'valid': (self.valid_df, self.valid_size),
                            'test': (self.test_df, self.test_size)}

        self.set_split('train')

    def set_split(self, split='train'):
        self._target_split = split
        self._target_df, self._target_size = self.lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, idx):
        row = self._target_df.iloc[idx]
        context = self.vectorizer.vectorize_context(row)
        target = self.vectorizer.vectorize_target(row)
        return {'idx': idx,
                'context': torch.tensor(context).to(self.device),
                'target': torch.tensor(target).to(self.device)}

    @classmethod
    def load_dataset_and_make_vectorizer(cls, text_df):
        print('input dataset: {:,} records'.format(len(text_df)))
        vectorizer = CBOWVectorizer.from_dataframe(text_df)
        return cls(text_df, vectorizer)