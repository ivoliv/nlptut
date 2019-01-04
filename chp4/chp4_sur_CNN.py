import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nlpClasses import Vocabulary

class SurnamesVectorizer(object):
    def __init__(self, surnames_vocab, nationality_vocab):
        self.surnames_vocab = surnames_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname, seq_len):
        one_hot = np.zeros((len(self.surnames_vocab), seq_len))
        for i, c in enumerate(surname):
            if i >= seq_len: break
            one_hot[self.surnames_vocab.lookup_token(c), i] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, surnames_df):
        surnames_vocab = Vocabulary(unk_token='@')
        nationality_vocab = Vocabulary(add_unk=False)

        for index, row in surnames_df.iterrows():
            for c in row.surname:
                surnames_vocab.add_token(c)
            nationality_vocab.add_token(row.nationality)

        return cls(surnames_vocab, nationality_vocab)


class SurnamesDataset(Dataset):
    def __init__(self, surnames_df, vectorizer):
        self.device = torch.device('cuda')\
            if torch.cuda.is_available() \
            else torch.device('cpu')
        self.surnames_df = surnames_df
        self._vectorizer = vectorizer
        
        self.max_surname_len = torch.max(torch.tensor(list(self.surnames_df['surname'].apply(lambda name: len(name))))).item()
        
        self.train_df = self.surnames_df[self.surnames_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.surnames_df[self.surnames_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.surnames_df[self.surnames_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.val_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    def set_split(self, split='train'):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        """ Return the size of the (selected) data set
        """
        return self._target_size

    def get_item(self, index):
        return self._target_df.iloc[index]

    def head(self, num=5):
        for i, row in enumerate(iter(self)):
            print(row)
            if i+1 >= num: break

    def input_size(self):
        return len(self._vectorizer.surnames_vocab)

    def output_size(self):
        return len(self._vectorizer.nationality_vocab)

    def __getitem__(self, index):
        """ Return the `index` item in the dataset
        In particular, from the indexed row, return:
        - surnames vector (x_data)
        - nationality index token (y_target)
        """

        row = self._target_df.iloc[index]

        surnames_vector = torch.tensor(self._vectorizer.vectorize(row.surname, 
                                                                  self.max_surname_len)).float()
        nationality_index = torch.tensor(self._vectorizer.nationality_vocab.lookup_token(row.nationality)).view(1)

        return {'idx': index,
                'x_data': surnames_vector.to(self.device),
                'y_target': nationality_index.to(self.device)}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def get_vectorizer(self):
        return self._vectorizer

    @classmethod
    def load_dataset_and_make_vectorizer(cls, filename):
        surnames_df = pd.read_csv(filename)
        return cls(surnames_df, SurnamesVectorizer.from_dataframe(surnames_df))



if __name__ == '__main__':

    filename = './data/data/names/full_dataset.csv'
    dataset = SurnamesDataset.load_dataset_and_make_vectorizer(filename)

    import random
    for i in range(2):
        idx = random.randint(0, len(dataset))
        print(dataset.__getitem__(idx))
