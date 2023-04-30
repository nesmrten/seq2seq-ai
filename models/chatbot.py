import torch
import json
from models.utils.tokenizer import Tokenizer
from models.seq2seq import EncoderRNN, DecoderRNN, Seq2Seq

class Chatbot:
    def __init__(self, model_file, tokenizer_file):
        self.tokenizer = Tokenizer.load(tokenizer_file)

        self.encoder = Encoder(len(self.tokenizer.vocab))
        self.decoder = Decoder(len(self.tokenizer.vocab))

        model_dict = torch.load(model_file, map_location=torch.device('cpu'))

        self.encoder.load_state_dict(model_dict['encoder_state_dict'])
        self.decoder.load_state_dict(model_dict['decoder_state_dict'])

        self.seq2seq = Seq2Seq(self.encoder, self.decoder)

    def generate_response(self, input_str):
        input_tkn = self.tokenizer.encode(input_str)
        input_len = torch.LongTensor([len(input_tkn)])
        input_tkn = torch.LongTensor(input_tkn).unsqueeze(0)

        enc_output, enc_hidden = self.seq2seq.encoder(input_tkn, input_len)
        dec_hidden = enc_hidden
        dec_input = torch.LongTensor([[self.tokenizer.vocab['<sos>']]])

        max_len = 20
        result_tkn = []
        for i in range(max_len):
            dec_output, dec_hidden = self.seq2seq.decoder(dec_input, dec_hidden)
            dec_output_tkn = dec_output.argmax(dim=1).item()
            result_tkn.append(dec_output_tkn)
            if dec_output_tkn == self.tokenizer.vocab['<eos>']:
                break
            dec_input = torch.LongTensor([[dec_output_tkn]])

        result_str = self.tokenizer.decode(result_tkn)
        return result_str
