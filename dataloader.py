import os
import tempfile
from typing import Union, List

import torch
import torch.nn as nn
from torch.utils import data

from constants import OUT_OF_VOCAB_TOKEN, PADDING_TOKEN, END_OF_SENTENCE_TOKEN
from util import read_lines_from_directory, download_file, decompress_tar_gz


class Encoder:
    def __init__(self):
        self._vocab = {}
        self._inverse_vocab = {}
        self._current_id = 0
        self._init_vocab()

    def _init_vocab(self) -> None:
        self.add_to_vocab(PADDING_TOKEN)
        self.add_to_vocab(END_OF_SENTENCE_TOKEN)
        self.add_to_vocab(OUT_OF_VOCAB_TOKEN)
        for i in range(32, 123):
            self.add_to_vocab(chr(i))

    def add_to_vocab(self, token: str) -> None:
        if token not in self._vocab:
            self._vocab[token] = self._current_id
            self._inverse_vocab[self._current_id] = token
            self._current_id += 1

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def get_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[OUT_OF_VOCAB_TOKEN])

    def get_token(self, id_: int) -> str:
        return self._inverse_vocab.get(id_, OUT_OF_VOCAB_TOKEN)

    def map_tokens_to_ids(self, tokens: Union[list, str]) -> list:
        return [self.get_id(token) for token in tokens]

    def map_ids_to_tokens(self, ids: List[int]) -> list:
        return [self.get_token(i) for i in ids]


class Dataset(data.Dataset):
    def __init__(self, dataset: str='imdb'):
        self.sentences = []
        self.encoder = Encoder()
        if dataset.lower() == 'imdb':
            self._load_imdb_dataset()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index], self.lengths[index]

    def _load_imdb_dataset(self):
        train_dir_path = os.path.join('aclImdb', 'train')
        if not os.path.exists(train_dir_path):
            imdb_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            _, tmp_file_path = tempfile.mkstemp(dir='.')
            print('Downloading IMDB dataset')
            download_file(imdb_url, tmp_file_path, 'wb')
            print('Dataset downloaded\nDecompressing the file')
            decompress_tar_gz(tmp_file_path, '.')
            os.remove(tmp_file_path)
        print('Loading and pre-processing data')
        for dir_path in ['pos', 'neg']:
            self.sentences.extend(read_lines_from_directory(os.path.join(train_dir_path, dir_path)))
        self.X = [torch.tensor(self.encoder.map_tokens_to_ids(sentence)) for sentence in self.sentences]
        # Sort the data for torch.nn.utils.rnn.pad_sequence
        # We need to pad the sequence in order to use DataLoader
        self.X.sort(key=len, reverse=True)
        self.y = [torch.cat((x_i[1:], torch.tensor([self.encoder.get_id(END_OF_SENTENCE_TOKEN)]))) for x_i in self.X]
        self.lengths = [len(x_i) for x_i in self.X]
        self.X = nn.utils.rnn.pad_sequence(self.X, batch_first=True, padding_value=self.encoder.get_id(PADDING_TOKEN))
        self.y = nn.utils.rnn.pad_sequence(self.y, batch_first=True, padding_value=self.encoder.get_id(PADDING_TOKEN))
