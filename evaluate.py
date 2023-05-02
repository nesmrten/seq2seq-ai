import torch
import argparse
from models.seq2seq import Seq2Seq
from models.dataset import ChatbotDataset
from models.utils.tokenizer import Tokenizer
from config import Config
import random


def evaluate(model = Seq2Seq(tokenizer.vocab_size, Config.EMBEDDING_SIZE, Config.HIDDEN_SIZE, Config.OUTPUT_SIZE, Config.NUM_LAYERS, dropout=0.5, device=torch.device('cpu'))
):
    model.eval()
    with torch.no_grad():
        # Tokenize the input sentence
        input_tokens = tokenizer.tokenize(sentence)

        # Convert the input sentence to a tensor
        input_tensor = torch.LongTensor(input_tokens).unsqueeze(0)

        # Move the input tensor to the device
        input_tensor = input_tensor.to(Config.DEVICE)

        # Initialize the hidden state
        hidden = None

        # Generate the response
        response = []
        for i in range(max_length):
            # Generate the output and the new hidden state
            output, hidden = model(input_tensor, [1], hidden)

            # Convert the output to a probability distribution
            output_dist = output.squeeze().div(Config.TEMPERATURE).exp()

            # Sample from the output distribution
            top_index = torch.multinomial(output_dist, 1)[0]

            # Convert the index to a token and add it to the response
            token = tokenizer.get_token(top_index.item())
            if token == Config.EOS_TOKEN:
                break
            response.append(token)

            # Add the token to the input tensor for the next iteration
            input_tensor = torch.LongTensor([top_index.item()]).unsqueeze(0).to(Config.DEVICE)

        # Convert the response to a string and return it
        return " ".join(response)

def predict(input_sentence, model, dataset, device, max_length=Config.MAX_LENGTH):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.LongTensor(dataset.tokenizer.encode(input_sentence)).unsqueeze(0).to(device)
        hidden, cell = model.encoder.init_hidden_cell(1)
        hidden, cell = hidden.to(device), cell.to(device)
        encoder_outputs, (hidden, cell) = model.encoder(input_tensor, (hidden, cell))

        # Create a tensor to hold the decoder input
        decoder_input = torch.LongTensor([[dataset.tokenizer.get_id(Config.SOS_TOKEN)]])

        # Initialize the decoder hidden state and cell state with the encoder final hidden state and cell state
        hidden, cell = hidden.to(device), cell.to(device)

        # Initialize the output sentence as an empty list
        output_sentence = []

        # Generate the response
        for i in range(max_length):
            decoder_output, hidden, cell = model.decoder(decoder_input.to(device), (hidden, cell), encoder_outputs)
            decoder_output = decoder_output.squeeze(1)
            decoder_output = decoder_output.argmax(dim=1)
            output_sentence.append(dataset.tokenizer.decode(decoder_output.item()))
            if output_sentence[-1] == Config.EOS_TOKEN:
                break
            decoder_input = decoder_output.unsqueeze(0)

        # Join the words into a sentence and return the sentence
        output_sentence = " ".join(output_sentence)
        return output_sentence

if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a Seq2Seq model for a chatbot.")
    parser.add_argument("data_dir", type=str, help="the directory containing the data")
    parser.add_argument("vocab_file", type=str, help="the vocabulary file")
    parser.add_argument("model_path", type=str, help="the path to the trained model")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="the batch size for inference")
    parser.add_argument("--max_length", type=int, default=Config.MAX_LENGTH, help="the maximum length of the output sequence")
    parser.add_argument("--temperature", type=float, default=Config.TEMPERATURE, help="the temperature for sampling from the output distribution")
    parser.add_argument("--random_seed", type=int, default=Config.RANDOM_SEED, help="the random seed for reproducibility")
    args = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Load the tokenizer and dataset
    tokenizer = Tokenizer(args.vocab_file)
    dataset = ChatbotDataset(args.data_dir, args.vocab_file)

    # Load the model
    model = Seq2Seq(tokenizer.vocab_size, Config.EMBEDDING_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.DROPOUT)
    model.load_state_dict(torch.load(args.model_path))
    model.to(Config.DEVICE)

    # Enter an interactive loop
    while True:
        # Get the user's input
        input_sentence = input("> ")
        if input_sentence.strip() == "":
            continue

        # Generate the response
        response = evaluate(model, tokenizer, input_sentence, max_length)
        
