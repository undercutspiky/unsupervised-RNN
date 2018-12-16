import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from constants import PADDING_TOKEN
from dataloader import Dataset
from model import CharRNN
from util import print_colored_text

np.set_printoptions(precision=4)


def collate_fn(data):
    """Sorts the input(the mini-batch) by length and truncates all the data points to the max length of the input"""
    sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
    max_len = sorted_data[0][2]
    x = torch.stack([x_i[:max_len] for x_i, y_i, l_i in sorted_data])
    y = torch.stack([y_i[:max_len] for x_i, y_i, l_i in sorted_data])
    l = [l_i for _, _, l_i in sorted_data]
    return x, y, l


def get_sentiments(model, x, threshold):
    """Get clusters for a single example. Note that the functions resets the hidden states of the model."""
    model.reset_intermediate_vars()
    sentiments = []
    for x_i in x:
        _ = model.forward(x_i.view(1, 1), take_log=False)
        sentiments.append(model.select_attn(model.attn, threshold))
    model.reset_intermediate_vars()
    return sentiments


def main():
    ds = Dataset('imdb')
    params = {'batch_size': 67,
              'shuffle': True,
              'num_workers': 8,
              'collate_fn': collate_fn}
    epochs = 4
    lr = 0.01
    tbptt_steps = 256
    training_generator = data.DataLoader(ds, **params)
    model = CharRNN(input_size=ds.encoder.get_vocab_size(),
                    embedding_size=8,
                    hidden_size=128,
                    output_size=ds.encoder.get_vocab_size(),
                    no_sentiments=3,
                    dense_size=32,
                    padding_idx=ds.encoder.get_id(PADDING_TOKEN),
                    n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_no = 0
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for x_i, y_i, l_i in training_generator:
            model.reset_intermediate_vars()
            step_no += 1
            print(x_i.size())
            batch_loss = 0
            for step in range(l_i[0] // tbptt_steps + (l_i[0] % tbptt_steps != 0)):
                von = tbptt_steps * step
                bis = min(tbptt_steps * (step + 1), l_i[0])
                out = model(x_i[:, von: bis])
                if step % 25 == 0:
                    print(model.attn[0].detach().numpy(), model.attn[-1].detach().numpy())
                loss = model.loss(out, y_i, l_i, von, bis)
                batch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                for p in model.parameters():
                    p.data.add_(-lr, p.grad.data)
                optimizer.step()

                model.detach_intermediate_vars()
            print('Total loss for this batch: ', batch_loss.item())
            if step_no % 30 == 1:
                gen_sample, sentis = model.generate_text(ds.encoder, 'T', 200, 0.7)
                print_colored_text(gen_sample, sentis, ds.encoder)
                # Print an example with sentiments
                print_colored_text(x_i[-1].data.numpy(), get_sentiments(model, x_i[-1], 0.7), ds.encoder)


if __name__ == '__main__':
    main()
