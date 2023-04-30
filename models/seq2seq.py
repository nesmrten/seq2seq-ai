import torch
import torch.nn as nn
from .encoder import EncoderRNN
from .decoder import DecoderRNN
from .attention import Attention



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else 20

        # Tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_size).to(self.device)

        # Encoder output
        enc_output, hidden = self.encoder(src)

        # Decoder input
        dec_input = torch.tensor([[self.decoder.sos_token_idx] * batch_size], device=self.device)

        # Decoder hidden state
        dec_hidden = hidden[:self.decoder.num_layers]

        # Decoding
        for t in range(1, max_len):
            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
            outputs[t] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = (trg[t].unsqueeze(0) if teacher_force else top1.unsqueeze(0))
        return outputs
