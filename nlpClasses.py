class Vocabulary(object):

    def __init__(self, token_to_index=None, add_unk=True, unk_token='<UNK>'):

        if token_to_index is None:
            token_to_index = {}
        self._token_to_index = token_to_index

        self._index_to_token = {idx: token for idx, token in self._token_to_index.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def add_token(self, token):

        if token not in self._token_to_index:
            token_index = len(self)
            self._token_to_index[token] = token_index
            self._index_to_token[token_index] = token
        else:
            token_index = self._token_to_index[token]

        return token_index

    def lookup_token(self, token):

        if self._unk_token:
            return self._token_to_index.get(token, self.unk_index)
        else:
            return self._token_to_index[token]

    def lookup_index(self, index):

        if index in self._index_to_token:
            return self._index_to_token[index]
        else:
            raise KeyError('the index {} is not in the vocabulary'.format(index))

    def __len__(self):
        return len(self._token_to_index)

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def to_serializable(self):
        return {'token_to_idx': self._token_to_index,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)