import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.rnn(x, hidden)
        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.rnn(x, hidden)
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_output, hidden = self.encoder(source, self.encoder.init_hidden(batch_size))

        x = target[:, 0]

        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            x = target[:, t] if torch.rand(1) < teacher_forcing_ratio else top1

        return outputs
