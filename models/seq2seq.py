import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import EncoderRNN
from .decoder import DecoderRNN
from .attention import Attention
from config import Config


class Seq2Seq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout, device):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderRNN(
            output_size, embedding_size, hidden_size, num_layers, dropout, Attention(hidden_size))
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.device = device
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

    # rest of the code


    def forward(self, input_tensor, input_lengths, target_tensor=None, teacher_forcing_ratio=0.5):
        # Embed the input tensor
        embedded = self.embedding(input_tensor)

        # Pack the embedded tensor into a PackedSequence
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)

        # Encode the packed tensor
        encoder_outputs, encoder_hidden = self.encoder(packed)

        # Create a tensor to hold the decoder input
        batch_size = input_tensor.size(0)
        decoder_input = torch.LongTensor([Config.SOS_TOKEN_ID] * batch_size).unsqueeze(1).to(self.device)

        # Initialize the decoder hidden state and cell state with the encoder final hidden state and cell state
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]

        # Use teacher forcing to train the model
        if target_tensor is not None:
            # Calculate the maximum target length
            max_target_length = max(input_lengths) + 2

            # Create a tensor to hold the decoder outputs
            decoder_outputs = torch.zeros(max_target_length, batch_size, self.decoder.output_size).to(self.device)

            # Use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                # Feed the target tensor through the decoder one timestep at a time
                for t in range(max_target_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_outputs[t] = decoder_output
                    decoder_input = target_tensor[:, t].unsqueeze(1)

            # Use the model's own predictions as inputs to the decoder
            else:
                # Feed the decoder's own previous output as the next input
                for t in range(max_target_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_outputs[t] = decoder_output
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach().unsqueeze(1)

            # Return the decoder outputs
            return decoder_outputs.transpose(0, 1).contiguous()

        # Use beam search to generate responses
        else:
            # Initialize the decoder output
            decoder_output = torch.LongTensor([Config.SOS_TOKEN_ID]).to(self.device)

            # Initialize the beam
            beam_size = Config.BEAM_SIZE
            min_length = Config.MIN_LENGTH
            max_length = Config.MAX_LENGTH
            eos_token_id = Config.EOS_TOKEN_ID
            vocab_size = Config.VOCAB_SIZE
            seq_len = 1
            beam_scores = torch.zeros(batch_size, beam_size, device=self.device)
            beam_scores[:, 1:] = float('-inf')
            beams = torch.zeros(batch_size, beam_size, max_length + 1, dtype=torch.long, device=self.device)
            beams[:, :, 0] = eos_token_id
            beams[:, :, 1] = Config.SOS_TOKEN_ID

            # Use beam search to generate responses
            for t in range(2, max_length + 1):
                # Create the decoder input
                decoder_input = beams[:, :, t - 1].clone().detach().long().unsqueeze(-1)

                # Decode the current timestep
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                # Calculate the log probabilities and scores
                log_probs = F.log_softmax(decoder_output, dim=1)
                scores = beam.scores.unsqueeze(-1) + log_probs

                # Apply length penalty
                length_penalty = ((5 + seq_len) / 6) ** Config.ALPHA
                scores /= length_penalty

                # Calculate the top scores and their indices
                top_scores, top_indices = scores.view(-1).topk(beam_size, largest=True, sorted=True)

                # Extract the sequences and their beam indices
                beam_idxs = top_indices // self.output_size
                token_idxs = top_indices % self.output_size
                prev_seq_idxs = beam_idxs // beam_size

                # Update the beam
                new_scores = top_scores
                new_seq_idxs = prev_seq_idxs
                new_token_idxs = token_idxs
                new_states = (decoder_hidden.unsqueeze(2).repeat(1, 1, beam_size, 1),
                            encoder_outputs.unsqueeze(1).repeat(1, beam_size, 1, 1))
                beam.update(new_scores, new_seq_idxs, new_token_idxs, new_states)

                # Check if all the sequences have ended
                if beam.is_done():
                    break

                # Update the decoder hidden state
                decoder_hidden = beam.get_last_hidden(encoder_outputs, decoder_hidden)

            # Return the top sequence
            return beam.get_top_sequence()
