import torch
import torch.nn as nn
from models.encoder import EncoderRNN
from models.decoder import DecoderRNN


class Seq2Seq:
    def __init__(self, model_file, tokenizer_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder, self.decoder = torch.load(model_file, map_location=self.device)
        self.tokenizer = self.load_tokenizer(tokenizer_file)

    def load_tokenizer(self, tokenizer_file):
        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        tokenizer = Tokenizer.from_json(tokenizer_json)
        return tokenizer

    def generate_response(self, user_input):
        with torch.no_grad():
            input_sequence = self.tokenizer.tokenize(user_input)
            input_tensor = self.tokenizer.convert_tokens_to_ids(input_sequence)
            input_tensor = torch.tensor(input_tensor, device=self.device).unsqueeze(0)

            encoder_hidden = self.encoder.init_hidden(1)
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([self.tokenizer.stoi['<start>']], device=self.device)
            decoded_tokens = []

            for i in range(self.tokenizer.max_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_output = decoder_output.squeeze(0)
                decoder_output = decoder_output.argmax(dim=0)
                decoded_tokens.append(decoder_output.item())
                if decoder_output.item() == self.tokenizer.stoi['<end>']:
                    break
                decoder_input = decoder_output.unsqueeze(0)

            response = self.tokenizer.convert_ids_to_tokens(decoded_tokens)
            response = ' '.join(response).replace('<start>', '').replace('<end>', '').strip()
            return response
