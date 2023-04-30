import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt




class LanguageIndex:
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


class Seq2SeqDataset(Dataset):
    def __init__(self, pairs, input_lang, target_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.target_lang = target_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text = self.pairs[idx][0]
        target_text = self.pairs[idx][1]

        input_tensor = self.tensorize(input_text, self.input_lang)
        target_tensor = self.tensorize(target_text, self.target_lang)

        return input_tensor, target_tensor

    def tensorize(self, sequence, lang):
        indexes = [lang.word2idx[word] for word in sequence.split(' ')]
        indexes.append(lang.word2idx['<pad>'])
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


def forward(self, input, hidden, encoder_output=None):
    """
    Forward pass through the decoder.

    Arguments:
    - input: A tensor of shape (batch_size, 1) representing the input tokens at each time step for the current batch
    - hidden: A tensor of shape (num_layers, batch_size, hidden_size) representing the hidden state of the decoder
    - encoder_output: A tensor of shape (max_length, batch_size, hidden_size) representing the output of the encoder at each time step for the current batch

    Returns:
    - output: A tensor of shape (batch_size, vocab_size) representing the scores for each vocabulary token at the current time step for the current batch
    - hidden: A tensor of shape (num_layers, batch_size, hidden_size) representing the updated hidden state of the decoder
    """
    # Embed the input token
    embedded = self.embedding(input).view(1, input.size(0), -1)

    # Combine the embedded input token and the context vector
    if encoder_output is not None:
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), dim=1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(
            1), encoder_output.transpose(0, 1))
        rnn_input = torch.cat((embedded[0], attn_applied.squeeze(1)), dim=1)
    else:
        rnn_input = embedded[0]

    # Pass the input and previous hidden state through the RNN
    output, hidden = self.rnn(rnn_input, hidden)

    # Calculate the scores for each vocabulary token
    output = self.out(output[0])

    return output, hidden
