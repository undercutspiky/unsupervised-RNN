from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataloader import Encoder
from util import coalesce


def init_rnn(rnn):
    for name, p in rnn.named_parameters():
        if name.startswith('weight_hh'):
            torch.nn.init.orthogonal_(p)
        elif name.startswith('weight_ih'):
            torch.nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            torch.nn.init.constant_(p, 1)


class SentiBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_layers: int=1):
        super(SentiBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=False)
        self.senti_dense = nn.Linear(hidden_size, output_size)
        init_rnn(self.rnn)

    def forward(self, x: torch.Tensor, hidden=None):
        """Expects a timestep as input and not the whole sequence"""
        batch_size, _ = x.size()
        hidden = hidden or self.init_hidden(batch_size)
        out, hidden = self.rnn(x.unsqueeze(0), hidden)
        out = out.view(batch_size, self.hidden_size)
        senti = self.senti_dense(out)
        senti = senti.view(batch_size, self.output_size)
        attn = F.softmax(senti, dim=1)
        return attn, hidden

    def init_hidden(self, batch_size: int):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))


class CharRNN(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, output_size: int, no_sentiments: int,
                 dense_size: int, padding_idx: int, n_layers: int=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.no_sentiments = no_sentiments

        self.embedder = nn.Embedding(num_embeddings=input_size,
                                     embedding_dim=embedding_size,
                                     padding_idx=padding_idx)
        self.rnn = nn.LSTM(input_size=no_sentiments + embedding_size,
                           hidden_size=hidden_size,
                           num_layers=n_layers,
                           batch_first=False)
        self.sentiment_block1 = SentiBlock(embedding_size, 8, no_sentiments)  # earlier: nn.Linear(input_size, 8)
        self.sentiment_block2 = SentiBlock(no_sentiments, 8, no_sentiments)  # earlier: nn.Linear(8, 3)
        self.dense = nn.Linear(hidden_size, dense_size)
        self.output_layer = nn.Linear(dense_size, output_size)
        self.loss_func = nn.NLLLoss()

        self.attn = None  # Represents softmax over sentiments
        self.hidden = None
        self.senti_hidden1 = None
        self.senti_hidden2 = None

        init_rnn(self.rnn)

    def forward(self, x, take_log=True):
        batch_size, seq_len = x.size()
        embeddings = self.embedder(x)
        # Convert from BxTxE to TxBxE to manually loop over time
        embeddings = torch.transpose(embeddings, 0, 1)
        # Initialize intermediate states
        self.hidden = coalesce(self.hidden, self.init_hidden(batch_size))
        self.attn = coalesce(self.attn, self.init_senti(batch_size))
        self.senti_hidden1 = coalesce(self.senti_hidden1, self.sentiment_block1.init_hidden(batch_size))
        self.senti_hidden2 = coalesce(self.senti_hidden2, self.sentiment_block2.init_hidden(batch_size))

        outputs = []
        for x_t in embeddings:
            out, self.hidden = self.rnn(torch.cat((x_t, self.attn), dim=1).unsqueeze(0), self.hidden)
            out = out.view(batch_size, self.hidden_size)
            out = self.dense(out)
            outputs.append(out)
            # For the next time step
            self.attn, self.senti_hidden1 = self.sentiment_block1(x_t)
            self.attn, self.senti_hidden2 = self.sentiment_block2(self.attn)
        outputs = torch.stack(outputs)
        outputs = outputs.view(-1, outputs.size(2))
        outputs = self.output_layer(outputs)
        if take_log:
            outputs = F.log_softmax(outputs, dim=1)
        outputs = outputs.view(seq_len, batch_size, self.output_size)
        return torch.transpose(outputs, 0, 1)

    def loss(self, predictions, y, lengths, von, bis):
        losses = []
        for i, j in enumerate(lengths):
            if von < min(bis, j):
                losses.append(self.loss_func(predictions[i, :min(bis, j) - von], y[i, von:min(bis, j)]))
        return torch.sum(torch.stack(losses))

    @staticmethod
    def sample(probabilities):
        return int(np.random.choice(len(probabilities), size=1, p=probabilities.data.numpy())[0])

    def generate_text(self, encoder: Encoder, starting_seq: Union[list, str], sample_size: int, threshold: float):
        def _single_fwd_pass(self, x, threshold):
            out = self.forward(torch.tensor(x).view(1, 1), take_log=False)
            out = out.view(self.output_size)
            out = F.softmax(out, dim=0)
            out = self.sample(out)
            return out, self.select_attn(self.attn, threshold)

        starting_symbols = encoder.map_tokens_to_ids(starting_seq)
        starting_symbols_tensor = torch.tensor(starting_symbols)
        self.reset_intermediate_vars()

        outputs = starting_symbols
        sentiments = [0]
        for x in starting_symbols_tensor:
            out, sentiment = _single_fwd_pass(self, x, threshold)
            sentiments.append(sentiment)
        outputs.append(out)

        for i in range(sample_size):
            out, sentiment = _single_fwd_pass(self, out, threshold)
            outputs.append(out)
            sentiments.append(sentiment)

        self.reset_intermediate_vars()
        return outputs, sentiments

    def select_attn(self, attn, threshold=0.5):
        attn = attn.squeeze()
        for i, score in enumerate(attn):
            if score > threshold:
                return i
        # If no sentiment has a score > threshold, pick another colour (and not color!)
        return self.no_sentiments

    def reset_intermediate_vars(self):
        self.attn = None
        self.hidden = None
        self.senti_hidden1 = None
        self.senti_hidden2 = None

    def detach_intermediate_vars(self):
        self.attn = self.attn.detach()
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.senti_hidden1 = (self.senti_hidden1[0].detach(), self.senti_hidden1[1].detach())
        self.senti_hidden2 = (self.senti_hidden2[0].detach(), self.senti_hidden2[1].detach())

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

    def init_senti(self, batch_size):
        return torch.zeros(batch_size, self.no_sentiments)
